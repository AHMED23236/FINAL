@echo off
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\msi\Desktop\merge _emna_ahmed"
"C:\Users\msi\Desktop\merge _emna_ahmed\cjo maintenance\.venv\Scripts\python.exe" daily_csv_generator.py >> "C:\Users\msi\Desktop\merge _emna_ahmed\cjo maintenance\cjo-maintenance-data-cleaned\daily\generator_log.txt" 2>&1
