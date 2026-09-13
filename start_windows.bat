@echo off
REM FlyChess : lance backend (FastAPI :8000) + frontend (statique :3000)
REM dans deux fenêtres séparées qui survivent à tout.
REM Double-clique ce fichier, puis ouvre http://localhost:3000



start "FlyChess backend" cmd /k "cd /d %~dp0backend && python -m uvicorn main:app --port 8000"
timeout /t 2 /nobreak >nul
start "FlyChess frontend" cmd /k "cd /d %~dp0frontend && python -m http.server 3000"
timeout /t 2 /nobreak >nul
start "" http://localhost:3000
echo.
echo  Backend  : http://localhost:8000  (fenetre "FlyChess backend")
echo  Frontend : http://localhost:3000  (fenetre "FlyChess frontend")
echo  Le cerveau met ~1-2 min a charger (26M synapses + mesure de reference).
echo  Ferme les deux fenetres cmd pour tout arreter.
