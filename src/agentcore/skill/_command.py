"""Bounded local shell execution for Skill tools."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Thread
from typing import BinaryIO

import anyio
from anyio.abc import ByteReceiveStream, Process

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


class _BoundedOutput:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._value = bytearray()
        self.truncated = False

    def append(self, chunk: bytes) -> None:
        remaining = self._limit - len(self._value)
        if remaining > 0:
            self._value.extend(chunk[:remaining])
        if len(chunk) > remaining:
            self.truncated = True

    def text(self) -> str:
        value = self._value.decode("utf-8", errors="replace")
        if not self.truncated:
            return value
        return f"{value}\n... [output truncated, exceeded {self._limit} bytes]"


async def run_command(
    command: str,
    *,
    cwd: str,
    timeout_seconds: int,
    output_limit: int,
) -> CommandResult:
    logger.info(
        "agentcore.skill.command.started mode=async cwd=%s timeout_seconds=%s",
        cwd,
        timeout_seconds,
    )
    stdout = _BoundedOutput(output_limit)
    stderr = _BoundedOutput(output_limit)
    process = await anyio.open_process(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        start_new_session=os.name == "posix",
    )
    try:
        with anyio.move_on_after(timeout_seconds) as timeout_scope:
            async with anyio.create_task_group() as tasks:
                assert process.stdout is not None
                assert process.stderr is not None
                tasks.start_soon(_drain_async, process.stdout, stdout)
                tasks.start_soon(_drain_async, process.stderr, stderr)
                await process.wait()
        if timeout_scope.cancel_called:
            with anyio.CancelScope(shield=True):
                await _terminate_async(process)
            result = CommandResult(
                stdout="",
                stderr=f"Command timed out after {timeout_seconds} seconds.",
                exit_code=-1,
                timed_out=True,
            )
            logger.warning(
                "agentcore.skill.command.completed mode=async cwd=%s exit_code=-1 timed_out=true",
                cwd,
            )
            return result
        assert process.returncode is not None
        result = CommandResult(
            stdout=stdout.text(),
            stderr=stderr.text(),
            exit_code=process.returncode,
            timed_out=False,
        )
        logger.info(
            "agentcore.skill.command.completed mode=async cwd=%s exit_code=%s timed_out=false",
            cwd,
            result.exit_code,
        )
        return result
    except BaseException:
        with anyio.CancelScope(shield=True):
            await _terminate_async(process)
        raise
    finally:
        with anyio.CancelScope(shield=True):
            await process.aclose()


def run_command_sync(
    command: str,
    *,
    cwd: str,
    timeout_seconds: int,
    output_limit: int,
) -> CommandResult:
    logger.info(
        "agentcore.skill.command.started mode=sync cwd=%s timeout_seconds=%s",
        cwd,
        timeout_seconds,
    )
    stdout = _BoundedOutput(output_limit)
    stderr = _BoundedOutput(output_limit)
    process = subprocess.Popen(  # noqa: S602
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        start_new_session=os.name == "posix",
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_done = Event()
    stderr_done = Event()
    readers = (
        Thread(
            target=_drain_sync,
            args=(process.stdout, stdout, stdout_done),
            daemon=True,
        ),
        Thread(
            target=_drain_sync,
            args=(process.stderr, stderr, stderr_done),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    while process.poll() is None or not stdout_done.is_set() or not stderr_done.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            _terminate_sync(process)
            break
        stdout_done.wait(min(remaining, 0.05))
    for reader in readers:
        reader.join(timeout=1)
    if process.poll() is None:
        _terminate_sync(process)
    if timed_out:
        result = CommandResult(
            stdout="",
            stderr=f"Command timed out after {timeout_seconds} seconds.",
            exit_code=-1,
            timed_out=True,
        )
        logger.warning(
            "agentcore.skill.command.completed mode=sync cwd=%s exit_code=-1 timed_out=true",
            cwd,
        )
        return result
    result = CommandResult(
        stdout=stdout.text(),
        stderr=stderr.text(),
        exit_code=process.returncode,
        timed_out=False,
    )
    logger.info(
        "agentcore.skill.command.completed mode=sync cwd=%s exit_code=%s timed_out=false",
        cwd,
        result.exit_code,
    )
    return result


async def _drain_async(stream: ByteReceiveStream, output: _BoundedOutput) -> None:
    async for chunk in stream:
        output.append(chunk)


def _drain_sync(stream: BinaryIO, output: _BoundedOutput, done: Event) -> None:
    try:
        while chunk := stream.read(64 * 1024):
            output.append(chunk)
    finally:
        done.set()


async def _terminate_async(process: Process) -> None:
    _signal_process_tree(process.pid, signal.SIGTERM, process.terminate)
    await anyio.sleep(0.1)
    _signal_process_tree(process.pid, getattr(signal, "SIGKILL", signal.SIGTERM), process.kill)
    with anyio.move_on_after(1):
        await process.wait()


def _terminate_sync(process: subprocess.Popen[bytes]) -> None:
    _signal_process_tree(process.pid, signal.SIGTERM, process.terminate)
    time.sleep(0.1)
    _signal_process_tree(process.pid, getattr(signal, "SIGKILL", signal.SIGTERM), process.kill)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _signal_process_tree(pid: int, signum: int, fallback: Callable[[], None]) -> None:
    try:
        if os.name == "posix":
            os.killpg(pid, signum)
        else:
            fallback()
    except (ProcessLookupError, PermissionError):
        try:
            fallback()
        except ProcessLookupError:
            pass
