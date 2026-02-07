# ═══════════════════════════════════════════════════════════════════════════════
# VICEROY 2026 V6 — TWO-HOP DEPLOYMENT SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════
# Run this from your laptop. You will be prompted for SSH credentials.
# Usage: .\deploy_to_tower2.ps1
# ═══════════════════════════════════════════════════════════════════════════════

$TOWER1 = "adrian@192.168.12.102"
$TOWER2 = "adrian@10.0.0.2"
$LOCAL_SCRIPT = "C:\Users\adria\Projects\VICEROY_2026_HDC_Sim\src\viceroy_hdc_v6_final.py"
$LOCAL_DATA = "C:\Users\adria\Projects\VICEROY_2026_HDC_Sim\.data\RML2016.10a_dict.pkl"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  VICEROY 2026 V6 — DEPLOYMENT INITIATED" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Create directories
# ─────────────────────────────────────────────────────────────────────────────
Write-Host "[PHASE 1] Creating directories on Tower 1 and Tower 2..." -ForegroundColor Yellow
ssh $TOWER1 "mkdir -p ~/viceroy/.data"
ssh $TOWER1 "ssh $TOWER2 'mkdir -p ~/viceroy/.data'"
Write-Host "[PHASE 1] ✓ Directories ready" -ForegroundColor Green
Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: HOP 1 — Laptop → Tower 1
# ─────────────────────────────────────────────────────────────────────────────
Write-Host "[PHASE 2] HOP 1: Transferring payload to Tower 1..." -ForegroundColor Yellow
Write-Host "  → Script (30 KB)..." -ForegroundColor Gray
scp $LOCAL_SCRIPT "${TOWER1}:~/viceroy/"
Write-Host "  → Dataset (611 MB) — this may take a few minutes..." -ForegroundColor Gray
scp $LOCAL_DATA "${TOWER1}:~/viceroy/.data/"
Write-Host "[PHASE 2] ✓ HOP 1 complete" -ForegroundColor Green
Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: HOP 2 — Tower 1 → Tower 2 (via SSH command)
# ─────────────────────────────────────────────────────────────────────────────
Write-Host "[PHASE 2] HOP 2: Transferring payload from Tower 1 → Tower 2..." -ForegroundColor Yellow
Write-Host "  → Script..." -ForegroundColor Gray
ssh $TOWER1 "scp ~/viceroy/viceroy_hdc_v6_final.py ${TOWER2}:~/viceroy/"
Write-Host "  → Dataset (611 MB) — this may take a few minutes..." -ForegroundColor Gray
ssh $TOWER1 "scp ~/viceroy/.data/RML2016.10a_dict.pkl ${TOWER2}:~/viceroy/.data/"
Write-Host "[PHASE 2] ✓ HOP 2 complete" -ForegroundColor Green
Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# DEPLOYMENT COMPLETE
# ─────────────────────────────────────────────────────────────────────────────
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ✓ PAYLOAD DEPLOYED TO TOWER 2" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "NEXT STEPS — Run experiments manually:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  ssh $TOWER1" -ForegroundColor White
Write-Host "  ssh $TOWER2" -ForegroundColor White
Write-Host "  cd ~/viceroy" -ForegroundColor White
Write-Host "  python3 viceroy_hdc_v6_final.py --dim 10000 --epochs 20 --output sniper_results.json" -ForegroundColor White
Write-Host "  python3 viceroy_hdc_v6_final.py --dim 2000 --epochs 5 --output scout_results.json" -ForegroundColor White
Write-Host ""
