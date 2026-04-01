param(
    [Parameter(Mandatory = $true)]
    [string]$experiment,

    [int]$sampleN = 4,
    [int]$seed = 42
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$bundleRoot = Join-Path $repoRoot "behaviour_stat_pipeline_review"

# Create folders
$dirs = @(
    "$bundleRoot",
    "$bundleRoot\key_code",
    "$bundleRoot\key_code\configs",
    "$bundleRoot\key_code\configs\experiments",
    "$bundleRoot\key_code\scripts",
    "$bundleRoot\key_code\src\behaviour_stat_pipeline\assays",
    "$bundleRoot\key_code\src\behaviour_stat_pipeline\common",
    "$bundleRoot\key_code\src\behaviour_stat_pipeline\stats\endpoints",
    "$bundleRoot\key_code\src\behaviour_stat_pipeline\figures",
    "$bundleRoot\tiny_test_data",
    "$bundleRoot\expected_outputs",
    "$bundleRoot\manual_spotchecks"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

# Files to mirror
$files = @(
    "configs\stats.yaml",
    "configs\schema_aliases.yaml",
    "configs\experiments\$experiment.yaml",
    "scripts\run_assay.py",
    "scripts\run_stats.py",
    "scripts\run_figures.py",
    "src\behaviour_stat_pipeline\assays\registry.py",
    "src\behaviour_stat_pipeline\assays\nor.py",
    "src\behaviour_stat_pipeline\assays\oft.py",
    "src\behaviour_stat_pipeline\assays\lm.py",
    "src\behaviour_stat_pipeline\assays\vit.py",
    "src\behaviour_stat_pipeline\assays\misc.py",
    "src\behaviour_stat_pipeline\common\schema.py",
    "src\behaviour_stat_pipeline\common\derive.py",
    "src\behaviour_stat_pipeline\common\qc.py",
    "src\behaviour_stat_pipeline\stats\endpoints\two_way_anova.py",
    "src\behaviour_stat_pipeline\stats\endpoints\posthoc_cellmeans_prism.py",
    "src\behaviour_stat_pipeline\stats\endpoints\one_sample.py",
    "src\behaviour_stat_pipeline\stats\endpoints\correlations.py",
    "src\behaviour_stat_pipeline\figures\runner.py",
    "src\behaviour_stat_pipeline\figures\boxpoints.py",
    "src\behaviour_stat_pipeline\figures\scatter.py"
)

foreach ($relPath in $files) {
    $src = Join-Path $repoRoot $relPath
    $dst = Join-Path $bundleRoot ("key_code\" + $relPath)

    if (Test-Path $src) {
        $dstDir = Split-Path -Parent $dst
        New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
        Copy-Item $src $dst -Force
        Write-Host "Copied $relPath"
    }
    else {
        Write-Warning "Missing: $relPath"
    }
}

Write-Host "Using Python:"
python --version
Write-Host "`nRebuilding tiny test data for $experiment ..."
python "$repoRoot\scripts\update_tiny_test_data.py" `
    --experiment $experiment `
    --n $sampleN `
    --seed $seed `
    --repo-root $repoRoot `
    --fresh

if ($LASTEXITCODE -ne 0) {
    throw "update_tiny_test_data.py failed."
}

# --- Rebuild zip ---
$zipPath = Join-Path $repoRoot "behaviour_stat_pipeline_review.$experiment.zip"

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Compress-Archive -Path $bundleRoot -DestinationPath $zipPath -Force
Write-Host "`nCreated $zipPath"

