# Reproduce on the authorized Windows Zeus host with the existing environment.
# No source, checkpoint, frozen-run or prior result writes.
$fixtureRoot = $PSScriptRoot
$fixtureRepo = 'C:\Users\David\dev\BCA'
$fixturePython = 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe'
$fixtureOutput = Join-Path $fixtureRoot ('rerun_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
& $fixturePython -B (Join-Path $fixtureRoot 'run_fixtures.py') --repo $fixtureRepo --output $fixtureOutput
$fixtureExit = $LASTEXITCODE
Write-Output "Fixture exit code: $fixtureExit. The recorded source has two scientific contract failures and is expected to exit 1. Inspect the JSON split between harness_status and contract_status."
# For a source-only bundle, change fixtureRepo to the bundled source root.
# If necessary, set BCA_SHIFT_DEPS to a compatible directory containing the
# exact dependency layout required by experiments/shift/__init__.py.
exit $fixtureExit
