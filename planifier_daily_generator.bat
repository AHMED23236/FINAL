@echo off
:: ============================================================
:: Planifie daily_csv_generator.py tous les jours à minuit
:: Exécuter EN TANT QU'ADMINISTRATEUR une seule fois
:: ============================================================

set PYTHON="C:\Users\msi\Desktop\merge _emna_ahmed\cjo maintenance\.venv\Scripts\python.exe"
set SCRIPT="C:\Users\msi\Desktop\merge _emna_ahmed\daily_csv_generator.py"
set TASK_NAME=CJO_Daily_Generator_Four

echo.
echo  [1] Suppression de l'ancienne tache (si existante)...
schtasks /delete /tn "%TASK_NAME%" /f 2>nul

echo  [2] Creation de la tache planifiee (tous les jours a 00:01)...
schtasks /create ^
  /tn "%TASK_NAME%" ^
  /tr "%PYTHON% %SCRIPT%" ^
  /sc DAILY ^
  /st 00:01 ^
  /ru "%USERNAME%" ^
  /rl HIGHEST ^
  /f

echo.
echo  [3] Verification...
schtasks /query /tn "%TASK_NAME%" /fo LIST

echo.
echo  ============================================================
echo   Tache planifiee : daily_csv_generator.py tous les jours a 00:01
echo   Le CSV du jour est genere dans :
echo   cjo maintenance\cjo-maintenance-data-cleaned\daily\
echo   Pour supprimer : schtasks /delete /tn %TASK_NAME% /f
echo  ============================================================
pause
