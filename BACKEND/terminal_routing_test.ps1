$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:8000"
$failures = 0
function Assert($name, $cond) {
    if ($cond) { Write-Host ("  PASS  " + $name) } else { Write-Host ("  FAIL  " + $name); $script:failures += 1 }
}

Write-Host "--- POST /api/claims/score"
$row = (Import-Csv "..\test_fixtures\test_uploaded_claims.csv")[0]
$features = @{}
$row.PSObject.Properties | ForEach-Object { $features[$_.Name] = $_.Value }
(@{ features = $features } | ConvertTo-Json -Depth 5) | Set-Content -Encoding utf8 claim_score_body.json
$claim = curl.exe -s -X POST "$base/api/claims/score" -H "Content-Type: application/json" --data-binary "@claim_score_body.json" | ConvertFrom-Json
Write-Host ("prediction={0} risk_score={1} calibrated_risk_score={2}" -f $claim.prediction, $claim.risk_score, $claim.calibrated_risk_score)
Write-Host ("routing={0}" -f ($claim.routing | ConvertTo-Json -Compress -Depth 4))
Assert "claim single has routing" ($claim.routing -ne $null)
Assert "claim single route is one of the three" (@("auto_approve","fast_track","full_investigation") -contains $claim.routing.route)
Assert "claim single has 7 audited predicates" ($claim.routing.predicate_results.PSObject.Properties.Name.Count -eq 7)
Assert "claim single reasons non-empty" ($claim.routing.reasons.Count -gt 0)

Write-Host "--- POST /api/providers/score"
$prow = (Import-Csv "..\test_fixtures\test_uploaded_providers.csv")[0]
$pfeatures = @{}
$prow.PSObject.Properties | ForEach-Object { $pfeatures[$_.Name] = $_.Value }
(@{ features = $pfeatures } | ConvertTo-Json -Depth 5) | Set-Content -Encoding utf8 provider_score_body.json
$prov = curl.exe -s -X POST "$base/api/providers/score" -H "Content-Type: application/json" --data-binary "@provider_score_body.json" | ConvertFrom-Json
Write-Host ("prediction={0} risk_score={1} calibrated_risk_score={2}" -f $prov.prediction, $prov.risk_score, $prov.calibrated_risk_score)
Write-Host ("routing={0}" -f ($prov.routing | ConvertTo-Json -Compress -Depth 4))
Assert "provider single has routing" ($prov.routing -ne $null)
Assert "provider single route is one of the three" (@("auto_approve","fast_track","full_investigation") -contains $prov.routing.route)

Write-Host "--- POST /api/claims/batch (multipart, real uploaded CSV)"
$cb = curl.exe -s -F "file=@..\test_fixtures\test_uploaded_claims.csv" "$base/api/claims/batch" | ConvertFrom-Json
$claimRoutes = $cb.results | ForEach-Object { $_.routing.route }
Write-Host ("total={0} results={1} every-row-routed={2}" -f $cb.total, $cb.results.Count, ((($cb.results | ForEach-Object { $_.routing -ne $null }) -notcontains $false)))
Write-Host ("route distribution: auto={0} fast={1} full={2}" -f (($claimRoutes | Where-Object { $_ -eq "auto_approve" }).Count), (($claimRoutes | Where-Object { $_ -eq "fast_track" }).Count), (($claimRoutes | Where-Object { $_ -eq "full_investigation" }).Count))
Write-Host ("summary.routing_counts={0}" -f ($cb.summary.routing_counts | ConvertTo-Json -Compress))
Write-Host ("queue top: {0} risk={1} (original ML risk, unchanged by routing)" -f $cb.queue[0].claim_id, $cb.queue[0].risk_score)
Assert "claim batch: 25 rows each with exactly one route" ($cb.results.Count -eq 25 -and (($cb.results | ForEach-Object { $_.routing -ne $null }) -notcontains $false))
Assert "claim batch: routing_counts sum == 25" ((($cb.summary.routing_counts.PSObject.Properties | ForEach-Object { $_.Value } | Measure-Object -Sum).Sum) -eq 25)
Assert "claim batch: queue has no routing keys" ((($cb.queue | ForEach-Object { $_.PSObject.Properties.Name -contains "route" }) -contains $true) -eq $false)

Write-Host "--- POST /api/providers/batch (multipart, real uploaded CSV)"
$pb = curl.exe -s -F "file=@..\test_fixtures\test_uploaded_providers.csv" "$base/api/providers/batch" | ConvertFrom-Json
$provRoutes = $pb.results | ForEach-Object { $_.routing.route }
Write-Host ("total={0} results={1} every-row-routed={2}" -f $pb.total, $pb.results.Count, ((($pb.results | ForEach-Object { $_.routing -ne $null }) -notcontains $false)))
Write-Host ("route distribution: auto={0} fast={1} full={2}" -f (($provRoutes | Where-Object { $_ -eq "auto_approve" }).Count), (($provRoutes | Where-Object { $_ -eq "fast_track" }).Count), (($provRoutes | Where-Object { $_ -eq "full_investigation" }).Count))
Write-Host ("summary.routing_counts={0}" -f ($pb.summary.routing_counts | ConvertTo-Json -Compress))
Write-Host ("queue top: {0} risk={1} (original ML risk, unchanged by routing)" -f $pb.queue[0].provider_npi, $pb.queue[0].risk_score)
Assert "provider batch: 7 rows each with exactly one route" ($pb.results.Count -eq 7 -and (($pb.results | ForEach-Object { $_.routing -ne $null }) -notcontains $false))
Assert "provider batch: routing_counts sum == 7" ((($pb.summary.routing_counts.PSObject.Properties | ForEach-Object { $_.Value } | Measure-Object -Sum).Sum) -eq 7)

Remove-Item claim_score_body.json, provider_score_body.json -ErrorAction SilentlyContinue
if ($failures -eq 0) { Write-Host "TERMINAL ROUTING API TEST: PASS" } else { Write-Host ("TERMINAL ROUTING API TEST: FAIL ({0})" -f $failures); exit 1 }
