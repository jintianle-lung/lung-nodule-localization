$ErrorActionPreference = "Stop"

$Root = "C:\Users\SWH\Desktop\GitHub_Docs_Package"
$Pkg = Join-Path $Root "deliverables\editor_submission_package_20260504"
$PkgResolved = (Resolve-Path -LiteralPath $Pkg).Path
$History = Join-Path $PkgResolved "99_history_full_dump_and_superseded_assets"
$MoveLog = Join-Path $History "_move_log.csv"

New-Item -ItemType Directory -Force -Path $History | Out-Null

$moves = @(
    @{
        Source = Join-Path $PkgResolved "02_all_available_svg_figures"
        Destination = Join-Path $History "full_svg_dump\02_all_available_svg_figures"
        Reason = "Full flattened SVG archive; preferred main/SI figures are curated in 07_preferred_main_and_si_figures."
    },
    @{
        Source = Join-Path $PkgResolved "03_source_data_and_tables"
        Destination = Join-Path $History "full_source_data_dump\03_source_data_and_tables"
        Reason = "Full flattened source-data dump; figure-level curated source data are in 08_curated_source_data_for_figures."
    },
    @{
        Source = Join-Path $PkgResolved "06_converted_or_wrapped_svg_assets"
        Destination = Join-Path $History "conversion_audit\06_converted_or_wrapped_svg_assets"
        Reason = "PNG/PDF conversion and wrapping audit; native/preferred SVG entry points remain in 07_preferred_main_and_si_figures."
    },
    @{
        Source = Join-Path $PkgResolved "05_captions_SI_references\references_seed.bib"
        Destination = Join-Path $History "superseded_reference_seed\references_seed.bib"
        Reason = "Superseded by references_seed_expanded.bib."
    },
    @{
        Source = Join-Path $PkgResolved "05_captions_SI_references\references_seed.md"
        Destination = Join-Path $History "superseded_reference_seed\references_seed.md"
        Reason = "Superseded by references_seed_expanded.md."
    }
)

$rows = New-Object System.Collections.Generic.List[object]

foreach ($move in $moves) {
    $src = $move.Source
    $dst = $move.Destination
    $status = "missing"
    $count = 0
    $bytes = 0

    if (Test-Path -LiteralPath $src) {
        $srcResolved = (Resolve-Path -LiteralPath $src).Path
        if (-not $srcResolved.StartsWith($PkgResolved, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to move path outside package: $srcResolved"
        }

        $dstParent = Split-Path -Parent $dst
        New-Item -ItemType Directory -Force -Path $dstParent | Out-Null

        if (Test-Path -LiteralPath $dst) {
            $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $dst = "$dst.moved_$stamp"
        }

        if ((Get-Item -LiteralPath $src).PSIsContainer) {
            $measure = Get-ChildItem -LiteralPath $src -Recurse -File | Measure-Object -Property Length -Sum
            $count = [int]$measure.Count
            $bytes = [int64]$measure.Sum
        } else {
            $item = Get-Item -LiteralPath $src
            $count = 1
            $bytes = [int64]$item.Length
        }

        Move-Item -LiteralPath $src -Destination $dst
        $status = "moved"
    }

    $rows.Add([pscustomobject]@{
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        source = $src
        destination = $dst
        status = $status
        file_count = $count
        bytes = $bytes
        reason = $move.Reason
    }) | Out-Null
}

$rows | Export-Csv -LiteralPath $MoveLog -NoTypeInformation -Encoding UTF8

$summary = [pscustomobject]@{
    updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    package = $PkgResolved
    history_folder = $History
    moved_items = ($rows | Where-Object { $_.status -eq "moved" }).Count
    missing_items = ($rows | Where-Object { $_.status -eq "missing" }).Count
    move_log = $MoveLog
}

$summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $History "_archive_summary.json") -Encoding UTF8
$summary | ConvertTo-Json -Depth 4
