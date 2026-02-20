# MnemoCore Integration Setup — Windows PowerShell
# =================================================
# One-command setup for Claude Code (MCP) on Windows.
# For full hook/wrapper support, use WSL or Git Bash.
#
# Usage:
#   .\setup.ps1
#   .\setup.ps1 -All
#   .\setup.ps1 -ClaudeCode

param(
    [switch]$All,
    [switch]$ClaudeCode,
    [switch]$Gemini,
    [switch]$Aider
)

$ErrorActionPreference = "Stop"

$ScriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$MnemoDir     = Split-Path -Parent $ScriptDir
$BridgePy     = Join-Path $ScriptDir "mnemo_bridge.py"
$ClaudeHome   = Join-Path $env:USERPROFILE ".claude"
$ClaudeMcp    = Join-Path $ClaudeHome "mcp.json"
$ClaudeSettings = Join-Path $ClaudeHome "settings.json"
$HooksDir     = Join-Path $ScriptDir "claude_code\hooks"

function Write-Info    { Write-Host "[INFO]  $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[OK]    $args" -ForegroundColor Green }
function Write-Warn    { Write-Host "[WARN]  $args" -ForegroundColor Yellow }
function Write-Err     { Write-Host "[ERROR] $args" -ForegroundColor Red }

# ── Prerequisite checks ────────────────────────────────────────────────────

Write-Info "Checking Python requests..."
$requestsCheck = python -c "import requests; print('ok')" 2>&1
if ($requestsCheck -ne "ok") {
    Write-Warn "Installing requests..."
    python -m pip install --quiet requests
}
Write-Success "Python requests available"

Write-Info "Checking MnemoCore connectivity..."
$healthCheck = python "$BridgePy" health 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Success "MnemoCore is online"
} else {
    Write-Warn "MnemoCore offline — start it first:"
    Write-Warn "  cd $MnemoDir"
    Write-Warn "  uvicorn mnemocore.api.main:app --port 8100"
}

# ── Claude Code MCP Setup ─────────────────────────────────────────────────

function Setup-ClaudeCode {
    Write-Info "Setting up Claude Code integration..."

    if (-not (Test-Path $ClaudeHome)) { New-Item -ItemType Directory -Path $ClaudeHome | Out-Null }
    New-Item -ItemType Directory -Path (Join-Path $ClaudeHome "mnemo_context") -Force | Out-Null

    # MCP config
    $McpTemplate = Get-Content (Join-Path $ScriptDir "claude_code\mcp_config.json") -Raw
    $McpTemplate = $McpTemplate `
        -replace '\$\{MNEMOCORE_DIR\}', $MnemoDir.Replace('\', '/') `
        -replace '\$\{HAIM_API_KEY\}', ($env:HAIM_API_KEY ?? '')

    if (-not (Test-Path $ClaudeMcp)) {
        '{"mcpServers":{}}' | Set-Content $ClaudeMcp
    }

    $Existing = Get-Content $ClaudeMcp -Raw | ConvertFrom-Json
    $New      = $McpTemplate | ConvertFrom-Json
    if (-not $Existing.mcpServers) { $Existing | Add-Member -MemberType NoteProperty -Name mcpServers -Value @{} }
    $New.mcpServers.PSObject.Properties | ForEach-Object {
        $Existing.mcpServers | Add-Member -MemberType NoteProperty -Name $_.Name -Value $_.Value -Force
    }
    $Existing | ConvertTo-Json -Depth 10 | Set-Content $ClaudeMcp
    Write-Success "MCP server registered in $ClaudeMcp"

    # Hooks
    if (-not (Test-Path $ClaudeSettings)) { '{}' | Set-Content $ClaudeSettings }
    $Settings = Get-Content $ClaudeSettings -Raw | ConvertFrom-Json

    if (-not $Settings.hooks) {
        $Settings | Add-Member -MemberType NoteProperty -Name hooks -Value @{}
    }
    $hooksObj = $Settings.hooks

    $preCmd  = "python `"$($HooksDir.Replace('\','/'))/pre_session_inject.py`""
    $postCmd = "python `"$($HooksDir.Replace('\','/'))/post_tool_store.py`""

    if (-not $hooksObj.PreToolUse) {
        $hooksObj | Add-Member -MemberType NoteProperty -Name PreToolUse -Value @()
    }
    if (-not $hooksObj.PostToolUse) {
        $hooksObj | Add-Member -MemberType NoteProperty -Name PostToolUse -Value @()
    }

    $existingPre = $hooksObj.PreToolUse | ForEach-Object { $_.hooks[0].command }
    if ($preCmd -notin $existingPre) {
        $hooksObj.PreToolUse += @{matcher=".*"; hooks=@(@{type="command"; command=$preCmd})}
    }
    $existingPost = $hooksObj.PostToolUse | ForEach-Object { $_.hooks[0].command }
    if ($postCmd -notin $existingPost) {
        $hooksObj.PostToolUse += @{matcher="Edit|Write|MultiEdit"; hooks=@(@{type="command"; command=$postCmd})}
    }

    $Settings | ConvertTo-Json -Depth 10 | Set-Content $ClaudeSettings
    Write-Success "Hooks installed in $ClaudeSettings"

    # CLAUDE.md snippet
    $ClaudeMd = Join-Path $MnemoDir "CLAUDE.md"
    $Snippet  = Get-Content (Join-Path $ScriptDir "claude_code\CLAUDE_memory_snippet.md") -Raw
    $Marker   = "# MnemoCore — Persistent Cognitive Memory"
    if (Test-Path $ClaudeMd) {
        $Current = Get-Content $ClaudeMd -Raw
        if ($Current -notlike "*$Marker*") {
            Add-Content $ClaudeMd "`n$Snippet"
            Write-Success "Memory instructions appended to CLAUDE.md"
        } else {
            Write-Info "CLAUDE.md already contains MnemoCore instructions"
        }
    } else {
        $Snippet | Set-Content $ClaudeMd
        Write-Success "Created CLAUDE.md with memory instructions"
    }

    Write-Success "Claude Code integration complete"
}

# ── Main ───────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║    MnemoCore Integration Setup (Win)    ║" -ForegroundColor Magenta
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""

if (-not ($All -or $ClaudeCode -or $Gemini -or $Aider)) {
    Write-Host "Choose integrations:"
    Write-Host "  1) Claude Code (MCP + hooks + CLAUDE.md) — recommended"
    Write-Host "  4) All"
    $choice = Read-Host "Enter choice"
    switch ($choice) {
        "1" { $ClaudeCode = $true }
        "4" { $All = $true }
    }
}

if ($All -or $ClaudeCode) { Setup-ClaudeCode }

Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   Setup complete!                       ║" -ForegroundColor Green
Write-Host "║                                          ║" -ForegroundColor Green
Write-Host "║   Test: python integrations/mnemo_bridge.py health" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
