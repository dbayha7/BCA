# Run this yourself in a PowerShell terminal. The existing WSL approval program
# asks you to type the fingerprint; this script never supplies that input.
[CmdletBinding()]
param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$repoPath = 'C:\Users\David\dev\BCA'
$helperPath = '/mnt/c/Users/David/dev/BCA/outputs/shift_hosts_v4/complete_study.py'
$transferRef = 'refs/heads/codex/shift-hosts-v4-transfer'
$targetRef = 'refs/heads/claude/shift-hosts'

function Invoke-StudyStep([string]$Mode) {
    & wsl -u root -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 `
        /opt/bca-venv/bin/python -B $helperPath $Mode
    if ($LASTEXITCODE -ne 0) { throw "Study step $Mode stopped (exit $LASTEXITCODE)." }
}

function Publish-StudyCommit {
    & git -C $repoPath push origin "${transferRef}:${targetRef}"
    if ($LASTEXITCODE -ne 0) { throw 'Publishing stopped. Nothing will start before the commits reach GitHub.' }
}

Invoke-StudyStep '--check'
if ($CheckOnly) { return }

# The seven package files are committed exactly as tested. The transfer ref avoids
# pushing into the branch checked out in Windows; its working files stay intact.
Invoke-StudyStep '--freeze'
Publish-StudyCommit
Invoke-StudyStep '--approve'
Invoke-StudyStep '--commit-approval'
Publish-StudyCommit
Invoke-StudyStep '--launch'

Write-Host 'The remaining experiments are running in WSL. Keep WSL running.'
Write-Host 'Results export automatically after completion; a stop rule writes a report.'
