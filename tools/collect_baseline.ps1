param([int]$FirstGroup = 6)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
Set-Location -LiteralPath $projectRoot
# Optional future collection only. This script never prepares data or trains.
# Keep the validated camera; vary only conservative baseline speed targets.
$profiles = @(@(0.85, 0.60), @(0.90, 0.65), @(0.95, 0.65), @(1.00, 0.70), @(0.90, 0.60))
New-Item -ItemType Directory -Path 'configs/collection' -Force | Out-Null
for ($index = 0; $index -lt $profiles.Count; $index++) {
    $groupName = 'auto_{0:D2}' -f ($FirstGroup + $index)
    foreach ($metaFile in Get-ChildItem -Path 'data/sessions/*/session.json') {
        $existing = Get-Content -LiteralPath $metaFile.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($existing.group_id -eq $groupName) { throw "Group $groupName already exists; choose a new FirstGroup." }
    }
    $configuration = Get-Content -LiteralPath 'configs/race.json' -Raw -Encoding UTF8 | ConvertFrom-Json
    $configuration.baseline.speed_straight = $profiles[$index][0]
    $configuration.baseline.speed_turn = $profiles[$index][1]
    $configPath = "configs/collection/$groupName.json"
    $configuration | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $configPath -Encoding UTF8
    $logRoot = "runs/collection/$groupName"
    & $pythonPath -m racing.drive --config $configPath --mode baseline --connect --record --baseline-validated --group $groupName --seconds 300 --snapshots 3 --logdir $logRoot
    if ($LASTEXITCODE -ne 0) { throw "Collection failed for $groupName; inspect its logs before continuing." }
    $runPath = (Get-ChildItem -LiteralPath $logRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    & $pythonPath -m racing.laps $runPath --plot
    if ($LASTEXITCODE -ne 0) { throw "Lap verification failed for $groupName." }
    $lapReport = Get-Content -LiteralPath (Join-Path $runPath 'lap_validation.json') -Raw | ConvertFrom-Json
    if ($lapReport.validated_laps -lt 3 -or $null -ne $lapReport.run_fault) {
        throw "Fewer than three verified laps or a run fault in $groupName; review the session."
    }
}
Write-Output 'Five autonomous sessions completed. Run racing.audit_data before preparing a new dataset version.'
