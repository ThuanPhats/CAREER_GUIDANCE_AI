@echo off
chcp 65001 >nul
title CAREER GUIDANCE AI — Launcher

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║       CAREER GUIDANCE AI - LAUNCHER      ║
echo  ╚══════════════════════════════════════════╝
echo.

:: ── 1. Locate Python ──────────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python khong tim thay. Hay cai Python va thu lai.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo  [OK] %%i
echo.

:: ── 2. Install / update requirements ─────────────────────────────────────────
if not exist "requirements.txt" (
    echo  [WARN] Khong tim thay requirements.txt — bo qua buoc cai module.
) else (
    echo  [1/2] Dang cai dat cac module tu requirements.txt ...
    python -m pip install --upgrade pip --quiet
    python -m pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo  [ERROR] Cai module that bai. Kiem tra requirements.txt va ket noi mang.
        pause
        exit /b 1
    )
    echo  [OK]  Tat ca module da duoc cai dat.
)
echo.

:: ── 3. Run app.py (which launches Streamlit) ─────────────────────────────────
echo  [2/2] Dang khoi dong ung dung ...
echo  [INFO] Truy cap: http://localhost:8501
echo.
python app.py

if errorlevel 1 (
    echo.
    echo  [ERROR] app.py gap loi. Kiem tra log ben tren.
    pause
    exit /b 1
)

pause
