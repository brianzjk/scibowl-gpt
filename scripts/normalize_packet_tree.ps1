param(
    [string]$PacketsRoot = "data\raw\packets"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Normalize-Token {
    param([string]$Value)
    $normalized = $Value.ToLowerInvariant()
    $normalized = [regex]::Replace($normalized, '[^a-z0-9]+', '_')
    $normalized = [regex]::Replace($normalized, '_+', '_')
    return $normalized.Trim('_')
}

function Normalize-FolderName {
    param([string]$Name)
    if ($Name -match '^([A-Za-z]+)(20\d{2})$') {
        return ("{0}_{1}" -f $matches[1].ToLowerInvariant(), $matches[2])
    }
    return (Normalize-Token $Name)
}

function Get-FolderSlug {
    param([System.IO.DirectoryInfo]$Directory)
    return (Normalize-Token $Directory.Name)
}

function Get-UniqueFilePath {
    param(
        [System.IO.DirectoryInfo]$Directory,
        [string]$BaseName,
        [string]$Extension
    )

    $candidate = Join-Path $Directory.FullName ($BaseName + $Extension)
    if (-not (Test-Path $candidate)) {
        return $candidate
    }

    $index = 2
    while ($true) {
        $altCandidate = Join-Path $Directory.FullName ("{0}__alt_{1:D2}{2}" -f $BaseName, $index, $Extension)
        if (-not (Test-Path $altCandidate)) {
            return $altCandidate
        }
        $index += 1
    }
}

function Get-CanonicalStem {
    param(
        [string]$FolderSlug,
        [string]$OriginalStem
    )

    $normalized = $OriginalStem.ToLowerInvariant()
    $normalized = [regex]::Replace($normalized, '[\-_]+', ' ')
    $normalized = [regex]::Replace($normalized, '\s+', ' ').Trim()

    if ($normalized -match 'seeding\s+round') {
        return "{0}__seeding_round" -f $FolderSlug
    }
    if ($normalized -match 'finals?\s*(\d+)') {
        $roundNumber = [int]$matches[1]
        return "{0}__finals_{1:D2}" -f $FolderSlug, $roundNumber
    }
    if ($normalized -match '\bde\s*(\d+)\b') {
        $roundNumber = [int]$matches[1]
        $suffix = ""
        if ($normalized -match 'breakpoint') {
            $suffix = "_breakpoint"
        }
        return "{0}__de_{1:D2}{2}" -f $FolderSlug, $roundNumber, $suffix
    }
    if ($normalized -match '\brr\s*(\d+)\b') {
        $roundNumber = [int]$matches[1]
        return "{0}__rr_{1:D2}" -f $FolderSlug, $roundNumber
    }
    if ($normalized -match '\bround\s*(\d+)\b') {
        $roundNumber = [int]$matches[1]
        return "{0}__round_{1:D2}" -f $FolderSlug, $roundNumber
    }
    if ($normalized -match '\bwriters?\b') {
        return "{0}__writers" -f $FolderSlug
    }

    return "{0}__{1}" -f $FolderSlug, (Normalize-Token $OriginalStem)
}

function Split-NsbSets {
    param([string]$Root)

    $sourceDir = Join-Path $Root "nsb_hs_sets"
    if (-not (Test-Path $sourceDir)) {
        return
    }

    Get-ChildItem $sourceDir -File -Filter *.pdf | ForEach-Object {
        if ($_.BaseName -match '^set_(\d+)__(.+)$') {
            $setNumber = [int]$matches[1]
            $tail = $matches[2]
            $destFolderName = "nsb_set_{0:D2}" -f $setNumber
            $destDir = Join-Path $Root $destFolderName
            New-Item -ItemType Directory -Force -Path $destDir | Out-Null
            $destName = "{0}__{1}.pdf" -f $destFolderName, $tail
            $destPath = Get-UniqueFilePath -Directory (Get-Item $destDir) -BaseName ([System.IO.Path]::GetFileNameWithoutExtension($destName)) -Extension ".pdf"
            Move-Item $_.FullName $destPath
        }
    }

    if (-not (Get-ChildItem $sourceDir -Force | Select-Object -First 1)) {
        Remove-Item $sourceDir
    }
}

function Rename-Folders {
    param([string]$Root)

    Get-ChildItem $Root -Directory | ForEach-Object {
        $targetName = Normalize-FolderName $_.Name
        if ($targetName -ne $_.Name) {
            $targetPath = Join-Path $Root $targetName
            if (-not (Test-Path $targetPath)) {
                Rename-Item $_.FullName $targetName
            }
        }
    }
}

function Convert-WordDocsToPdf {
    param([string]$Root)

    $docFiles = Get-ChildItem $Root -Recurse -File -Include *.doc,*.docx
    if (-not $docFiles) {
        return @{
            Converted = 0
            Deleted = 0
            Failed = 0
        }
    }

    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $converted = 0
    $deleted = 0
    $failed = 0

    try {
        foreach ($doc in $docFiles) {
            $document = $null
            $pdfPath = [System.IO.Path]::ChangeExtension($doc.FullName, ".pdf")
            if (-not (Test-Path $pdfPath)) {
                try {
                    $document = $word.Documents.Open($doc.FullName)
                    $document.ExportAsFixedFormat($pdfPath, 17)
                    $document.Close()
                    $converted += 1
                }
                catch {
                    Write-Warning ("Failed to convert {0}: {1}" -f $doc.FullName, $_.Exception.Message)
                    $failed += 1
                    if ($document) {
                        $document.Close()
                    }
                    continue
                }
            }

            if (Test-Path $pdfPath) {
                Remove-Item $doc.FullName -Force
                $deleted += 1
            }
        }
    }
    finally {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
        [gc]::Collect()
        [gc]::WaitForPendingFinalizers()
    }

    return @{
        Converted = $converted
        Deleted = $deleted
        Failed = $failed
    }
}

function Rename-PacketFiles {
    param([string]$Root)

    Get-ChildItem $Root -Directory | ForEach-Object {
        $folder = $_
        $folderSlug = Get-FolderSlug $folder

        Get-ChildItem $folder.FullName -File -Filter *.pdf | ForEach-Object {
            $canonicalStem = Get-CanonicalStem -FolderSlug $folderSlug -OriginalStem $_.BaseName
            $directTarget = Join-Path $folder.FullName ($canonicalStem + ".pdf")
            if ($_.FullName -eq $directTarget) {
                return
            }

            if (Test-Path $directTarget) {
                $targetPath = Get-UniqueFilePath -Directory $folder -BaseName $canonicalStem -Extension ".pdf"
            }
            else {
                $targetPath = $directTarget
            }

            if ($_.FullName -ne $targetPath) {
                Move-Item $_.FullName $targetPath
            }
        }
    }
}

$rootPath = (Resolve-Path $PacketsRoot).Path

Split-NsbSets -Root $rootPath
Rename-Folders -Root $rootPath
$conversionStats = Convert-WordDocsToPdf -Root $rootPath
Rename-PacketFiles -Root $rootPath

Write-Host ("Word conversions: converted={0}, deleted={1}, failed={2}" -f $conversionStats.Converted, $conversionStats.Deleted, $conversionStats.Failed)
Get-ChildItem $rootPath -Directory | Select-Object Name
