@echo off
REM Double-click to upload this folder to https://github.com/swtail/jhss-debate-rhetoric
REM Requires Git installed. A GitHub sign-in window may pop up the first time — approve it.

cd /d "%~dp0"
echo Initializing repository in: %cd%
git --version || (echo Git is not installed. Install from https://git-scm.com/download/win and re-run. & pause & exit /b)

git init
git add .
git -c user.name="Bryan Kyung and Katie Kim" -c user.email="hangsoo.kyung@gmail.com" commit -m "Replication: validated multi-classifier re-analysis of debate rhetoric"
git branch -M main
git remote remove origin 2>nul
git remote add origin https://github.com/swtail/jhss-debate-rhetoric.git

echo.
echo Pushing to GitHub (this REPLACES the current repo contents with the complete version)...
echo Approve the sign-in window if it appears.
git push -u origin main --force

echo.
echo If push 