"""CLI оркестратор для управления микросервисами Titan Oracle."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

# Избегаем UnicodeEncodeError на Windows
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Пути к логам и состоянию
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

STATE_DIR = PROJECT_ROOT / "state"
STATE_DIR.mkdir(exist_ok=True)

RUNNING_PIDS_FILE = STATE_DIR / "running_pids.json"

# Сервисы
SERVICES = {
    "server": {
        "args": [
            sys.executable,
            "-m",
            "uvicorn",
            "btc_oracle.services.server.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        "color": "green",
    },
    "collector": {
        "args": [
            sys.executable,
            "-u",
            str(Path(__file__).parent / "services" / "collector" / "main.py"),
        ],
        "color": "blue",
    },
    "inferencer": {
        "args": [
            sys.executable,
            "-u",
            str(Path(__file__).parent / "services" / "inferencer" / "main.py"),
        ],
        "color": "magenta",
    },
    "trainer": {
        "args": [
            sys.executable,
            "-u",
            str(Path(__file__).parent / "services" / "trainer" / "main.py"),
        ],
        "color": "yellow",
    },
}

PROCESSES = {}


def _write_running_pids(pids: dict) -> None:
    """Записать PID'ы запущенных процессов."""
    STATE_DIR.mkdir(exist_ok=True)
    with open(RUNNING_PIDS_FILE, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.time(), "pids": pids}, f, indent=2)


def _load_running_pids() -> dict:
    """Загрузить PID'ы запущенных процессов."""
    try:
        if RUNNING_PIDS_FILE.exists():
            with open(RUNNING_PIDS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            return dict(data.get("pids") or {})
    except Exception:
        pass
    return {}


def _terminate_pid(pid: int) -> bool:
    """Остановить процесс по PID."""
    try:
        if os.name == "nt":
            r = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return r.returncode == 0
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
            return True
    except Exception:
        return False


def _wait_http_ok(url: str, *, timeout_s: float = 15.0, proc=None) -> bool:
    """Проверка готовности HTTP сервиса."""
    import urllib.request
    import urllib.error
    
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.time() + float(timeout_s)
    
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with opener.open(url, timeout=2) as resp:
                if 200 <= int(getattr(resp, "status", 0) or 0) < 300:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


@click.group()
def cli():
    """Titan Oracle Management CLI"""
    pass


@cli.command()
@click.option("--detach", is_flag=True, help="Запустить в фоне")
@click.option("--non-interactive", is_flag=True, help="Автоматический режим")
@click.option("--force-train", is_flag=True, help="Принудительное переобучение")
def bootstrap(detach: bool, non_interactive: bool, force_train: bool):
    """
    Полный запуск Titan Oracle:
    
    1. Проверка подключения к Bybit
    2. Инициализация БД
    3. Загрузка истории (если нужно)
    4. Запуск всех сервисов
    """
    console.print(Panel("Titan Oracle Bootstrap", style="bold cyan"))
    
    # Проверка Docker/Postgres
    console.print("\n[cyan]Шаг 1/4: Проверка базы данных...[/cyan]")
    if not _check_database():
        console.print("[yellow]⚠️ Запускаю Docker...[/yellow]")
        subprocess.run(["docker-compose", "up", "-d"], cwd=PROJECT_ROOT)
        time.sleep(5)
        
        if not _check_database():
            console.print("[red]❌ Ошибка подключения к БД[/red]")
            return
    
    console.print("[green]✅ База данных доступна[/green]")
    
    # Запуск UI Server первым
    console.print("\n[cyan]Шаг 2/4: Запуск UI Dashboard...[/cyan]")
    server_log = open(LOG_DIR / "server.log", "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        SERVICES["server"]["args"],
        cwd=PROJECT_ROOT,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    server_log.close()
    
    if not _wait_http_ok("http://127.0.0.1:8000/health", timeout_s=15.0, proc=server_proc):
        console.print("[red]❌ UI Server не запустился[/red]")
        return
    
    console.print("[green]✅ UI Dashboard: http://localhost:8000[/green]")
    PROCESSES["server"] = server_proc
    
    # Запуск остальных сервисов
    console.print("\n[cyan]Шаг 3/4: Запуск сервисов...[/cyan]")
    
    for name in ["collector", "inferencer", "trainer"]:
        log_path = LOG_DIR / f"{name}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            SERVICES[name]["args"],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        log_file.close()
        PROCESSES[name] = proc
        console.print(f"[green]✅ {name.capitalize()} started (PID: {proc.pid})[/green]")
        time.sleep(1)
        
        if proc.poll() is not None:
            console.print(f"[red]❌ {name.capitalize()} завершился. Проверьте logs/{name}.log[/red]")
            return
    
    console.print("\n[cyan]Шаг 4/4: Сохранение PID'ов...[/cyan]")
    pids = {"server": int(server_proc.pid)}
    for n, p in PROCESSES.items():
        if n != "server":
            pids[n] = int(p.pid)
    _write_running_pids(pids)
    
    console.print(Panel(
        "Bootstrap завершен!\n\n"
        "UI Dashboard: http://localhost:8000\n"
        "Сервисы работают в фоне",
        style="bold green"
    ))
    
    if detach:
        console.print("\n[green]Сервисы оставлены работать в фоне.[/green]")
        console.print("[dim]Остановить: titan stop[/dim]")
        return
    
    console.print("\n[yellow]Нажмите Ctrl+C для остановки...[/yellow]\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Остановка сервисов...[/yellow]")
        _stop_services()


@cli.command()
def start():
    """Запуск всех сервисов (без bootstrap проверок)."""
    console.print(Panel("🚀 Запуск Titan Oracle", style="bold green"))
    
    for name, conf in SERVICES.items():
        log_file = open(LOG_DIR / f"{name}.log", "w", encoding="utf-8")
        try:
            p = subprocess.Popen(
                conf["args"],
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            PROCESSES[name] = p
            console.print(f"✅ {name.capitalize()} started (PID: {p.pid}) -> logs/{name}.log")
            time.sleep(0.8)
        except FileNotFoundError as e:
            console.print(f"[red]Failed to start {name}: {e}[/red]")
        finally:
            log_file.close()
    
    pids = {name: int(p.pid) for name, p in PROCESSES.items()}
    _write_running_pids(pids)
    
    console.print(Panel("System is LIVE. (Ctrl+C to stop)", style="bold yellow"))
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        _stop_services()


@cli.command()
def stop():
    """Остановить все сервисы."""
    pids = _load_running_pids()
    
    if not pids:
        console.print("[yellow]Нет записанных PID'ов[/yellow]")
        return
    
    console.print("[yellow]Останавливаю сервисы...[/yellow]")
    for name, pid in pids.items():
        ok = _terminate_pid(pid)
        console.print(f"{'✅' if ok else '⚠️'} {name}: PID {pid}")
    
    try:
        RUNNING_PIDS_FILE.unlink()
    except Exception:
        pass


@cli.command()
def status():
    """Показать статус системы."""
    console.print(Panel("Titan Oracle Status", style="bold cyan"))
    
    # Database
    db_ok = _check_database()
    console.print(f"[{'green' if db_ok else 'red'}]Database: {'OK' if db_ok else 'NOT CONNECTED'}[/]")
    
    # Services
    pids = _load_running_pids()
    if pids:
        console.print("\n[bold]Running Services:[/bold]")
        for name, pid in pids.items():
            # Проверяем, жив ли процесс
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["tasklist", "/FI", f"PID eq {pid}"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True,
                    )
                    status = "running"
                else:
                    os.kill(pid, 0)  # Проверка существования
                    status = "running"
            except Exception:
                status = "stopped"
            
            console.print(f"  {name}: PID {pid} [{status}]")
    else:
        console.print("\n[yellow]No running services[/yellow]")


@cli.command()
@click.option("--yes", "-y", is_flag=True, help="Не спрашивать подтверждение")
def reset(yes: bool):
    """Очистка логов и состояния."""
    if not yes:
        click.confirm("Очистить логи и state?", abort=True)
    
    # Очистка логов
    cleaned = 0
    for folder in [LOG_DIR, STATE_DIR]:
        if folder.exists():
            for item in folder.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                        cleaned += 1
                    elif item.is_dir():
                        import shutil
                        shutil.rmtree(item)
                        cleaned += 1
                except Exception:
                    pass
    
    console.print(f"[green]✅ Reset complete: cleaned {cleaned} item(s)[/green]")


def _check_database() -> bool:
    """Проверка доступности БД."""
    try:
        import asyncio
        from btc_oracle.db import AsyncSessionLocal
        from sqlalchemy import text
        
        async def test_connection():
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
                return True
        
        return asyncio.run(test_connection())
    except Exception:
        return False


def _stop_services():
    """Остановить все запущенные сервисы."""
    for name, p in PROCESSES.items():
        try:
            p.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    cli()
