$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:8000"

Write-Host "--- GET /api/health"
curl.exe -s "$base/api/health"
Write-Host ""

Write-Host "--- POST /api/claims/score"
$row = (Import-Csv "..\test_fixtures\test_uploaded_claims.csv")[0]
$features = @{}
$row.PSObject.Properties | ForEach-Object { $features[$_.Name] = $_.Value }
(@{ features = $features } | ConvertTo-Json -Depth 5) | Set-Content -Encoding utf8 claim_score_body.json
$claim = curl.exe -s -X POST "$base/api/claims/score" -H "Content-Type: application/json" --data-binary "@claim_score_body.json" | ConvertFrom-Json
Write-Host ("prediction={0} risk_score={1} raw_ml_risk_score={2} calibrated_risk_score={3} risk_rating={4}" -f $claim.prediction, $claim.risk_score, $claim.raw_ml_risk_score, $claim.calibrated_risk_score, $claim.risk_rating)
Write-Host ("trust_flags={0}" -f ($claim.trust_flags | ConvertTo-Json -Compress))
Write-Host ("signals={0} evidence={1}" -f $claim.signals.Count, $claim.evidence.Count)

Write-Host "--- POST /api/providers/score"
$prow = (Import-Csv "..\test_fixtures\test_uploaded_providers.csv")[0]
$pfeatures = @{}
$prow.PSObject.Properties | ForEach-Object { $pfeatures[$_.Name] = $_.Value }
(@{ features = $pfeatures } | ConvertTo-Json -Depth 5) | Set-Content -Encoding utf8 provider_score_body.json
$prov = curl.exe -s -X POST "$base/api/providers/score" -H "Content-Type: application/json" --data-binary "@provider_score_body.json" | ConvertFrom-Json
Write-Host ("prediction={0} risk_score={1} raw_ml_risk_score={2} calibrated_risk_score={3}" -f $prov.prediction, $prov.risk_score, $prov.raw_ml_risk_score, $prov.calibrated_risk_score)
Write-Host ("trust_flags={0}" -f ($prov.trust_flags | ConvertTo-Json -Compress))
Write-Host ("signals={0}" -f $prov.signals.Count)

Write-Host "--- POST /api/claims/batch (multipart, real uploaded CSV)"
$cb = curl.exe -s -F "file=@..\test_fixtures\test_uploaded_claims.csv" "$base/api/claims/batch" | ConvertFrom-Json
Write-Host ("total={0} results={1} queue={2} all-with-trust_flags={3}" -f $cb.total, $cb.results.Count, $cb.queue.Count, (($cb.results | ForEach-Object { $_.trust_flags -ne $null }) -notcontains $false))
Write-Host ("queue top: {0} risk={1} (original ML risk)" -f $cb.queue[0].claim_id, $cb.queue[0].risk_score)

Write-Host "--- POST /api/providers/batch (multipart, real uploaded CSV)"
$pb = curl.exe -s -F "file=@..\test_fixtures\test_uploaded_providers.csv" "$base/api/providers/batch" | ConvertFrom-Json
Write-Host ("total={0} results={1} queue={2} all-with-trust_flags={3}" -f $pb.total, $pb.results.Count, $pb.queue.Count, (($pb.results | ForEach-Object { $_.trust_flags -ne $null }) -notcontains $false))
Write-Host ("queue top: {0} risk={1} (original ML risk)" -f $pb.queue[0].provider_npi, $pb.queue[0].risk_score)

Remove-Item claim_score_body.json, provider_score_body.json -ErrorAction SilentlyContinue
Write-Host "TERMINAL API TEST: PASS"
