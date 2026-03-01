param(
    [switch]$DoPush,
    [string]$RemoteName = "v2-private",
    [string]$ExpectedRepoUrl = "https://github.com/RobinALG87/MnemoCore-Persistent-Cognitive-Ai-Memory.git",
    [string]$ExpectedVersion = "2.0.0",
    [string]$TagName = "",
    [switch]$AllowDirty,
    [switch]$AllowVersionMismatch
)

$ErrorActionPreference = "Stop"

function Get-CurrentBranch {
    $branch = (& git branch --show-current).Trim()
    if (-not $branch) { throw "Could not determine current branch." }
    return $branch
}

function Ensure-Remote {
    param(
        [string]$Name,
        [string]$Url
    )

    $remoteNames = (& git remote)
    if (-not ($remoteNames -contains $Name)) {
        Write-Host "Remote '$Name' saknas, skapar..." -ForegroundColor Yellow
        & git remote add $Name $Url | Out-Null
    }

    $currentUrl = (& git remote get-url $Name).Trim()
    if ($currentUrl -ne $Url) {
        Write-Host "Remote URL mismatch. Uppdaterar $Name -> $Url" -ForegroundColor Yellow
        & git remote set-url $Name $Url | Out-Null
    }

    & git config --local "remote.$Name.pushurl" $Url | Out-Null
    & git config --local remote.pushDefault $Name | Out-Null
}

function Assert-CleanWorkingTree {
    param([bool]$AllowDirtyTree)
    if ($AllowDirtyTree) { return }

    $status = (& git status --porcelain)
    if ($status) {
        throw "Working tree är inte clean. Commit/stasha först eller kör med -AllowDirty."
    }
}

function Assert-Version {
    param(
        [string]$Expected,
        [bool]$AllowMismatch
    )

    $pyproject = Get-Content "pyproject.toml" -Raw
    if ($pyproject -notmatch 'version\s*=\s*"([^"]+)"') {
        if ($AllowMismatch) { return }
        throw "Kunde inte läsa version från pyproject.toml"
    }

    $version = $Matches[1]
    if ($version -ne $Expected -and -not $AllowMismatch) {
        throw "Version mismatch: pyproject.toml=$version men förväntad=$Expected"
    }
}

Write-Host "[v2-safe] Preflight startar..." -ForegroundColor Cyan

$branch = Get-CurrentBranch
Ensure-Remote -Name $RemoteName -Url $ExpectedRepoUrl
Assert-CleanWorkingTree -AllowDirtyTree:$AllowDirty.IsPresent
Assert-Version -Expected $ExpectedVersion -AllowMismatch:$AllowVersionMismatch.IsPresent

$remotePushUrl = (& git config --local --get "remote.$RemoteName.pushurl").Trim()
$pushDefault = (& git config --local --get remote.pushDefault).Trim()

Write-Host "[ok] Branch: $branch" -ForegroundColor Green
Write-Host "[ok] pushDefault: $pushDefault" -ForegroundColor Green
Write-Host "[ok] remote: $RemoteName" -ForegroundColor Green
Write-Host "[ok] pushurl: $remotePushUrl" -ForegroundColor Green
Write-Host "[ok] expected version: $ExpectedVersion" -ForegroundColor Green

if (-not $DoPush) {
    Write-Host "[dry-run] Ingen push utförd. Kör med -DoPush när ni är redo." -ForegroundColor Yellow
    Write-Host "Exempel: ./scripts/ops/push_v2_safe.ps1 -DoPush" -ForegroundColor Yellow
    exit 0
}

Write-Host "[push] Pusha branch '$branch' till '$RemoteName'..." -ForegroundColor Cyan
& git push $RemoteName "$branch"

if ($LASTEXITCODE -ne 0) {
    throw "Push av branch misslyckades."
}

if ($TagName) {
    Write-Host "[push] Pusha tag '$TagName'..." -ForegroundColor Cyan
    & git push $RemoteName "$TagName"
    if ($LASTEXITCODE -ne 0) {
        throw "Push av tag misslyckades."
    }
}

Write-Host "[done] Push klar." -ForegroundColor Green
