import json

# Carica i risultati
with open('outputs/runs/2026-02-18_noK3_noLeakage/metrics.json') as f:
    internal = json.load(f)

with open('outputs/runs/2026-02-18_noK3_noLeakage_external/metrics.json') as f:
    external = json.load(f)

print("\n" + "="*80)
print("CONFRONTO RAPIDO: Test Interno vs Esterno (2026-02-18)")
print("="*80)

print("\n📊 METRICHE PRINCIPALI:")
print(f"  Test Interno:  F1={internal['test_metrics']['f1']:.4f}, Prec={internal['test_metrics']['prec']:.4f}, Rec={internal['test_metrics']['rec']:.4f}")
print(f"  Test Esterno:  F1={external['test_metrics']['f1']:.4f}, Prec={external['test_metrics']['prec']:.4f}, Rec={external['test_metrics']['rec']:.4f}")

print("\n📉 PERFORMANCE DROP:")
f1_drop = (external['test_metrics']['f1'] - internal['test_metrics']['f1']) / internal['test_metrics']['f1'] * 100
prec_drop = (external['test_metrics']['prec'] - internal['test_metrics']['prec']) / internal['test_metrics']['prec'] * 100
rec_drop = (external['test_metrics']['rec'] - internal['test_metrics']['rec']) / internal['test_metrics']['rec'] * 100

print(f"  F1:        {f1_drop:+.1f}% {'🚨' if abs(f1_drop) > 20 else '⚠️' if abs(f1_drop) > 10 else '→'}")
print(f"  Precision: {prec_drop:+.1f}% {'🚨' if abs(prec_drop) > 20 else '⚠️' if abs(prec_drop) > 10 else '→'}")
print(f"  Recall:    {rec_drop:+.1f}% {'🚨' if abs(rec_drop) > 20 else '⚠️' if abs(rec_drop) > 10 else '→'}")

print("\n🎯 CONFUSION MATRIX:")
int_cm = internal['confusion_matrix']
ext_cm = external['confusion_matrix']

print(f"\n  Test Interno ({sum(sum(row) for row in int_cm)} samples):")
print(f"    TN={int_cm[0][0]:3d}  FP={int_cm[0][1]:3d}  (FP rate: {int_cm[0][1]/(int_cm[0][0]+int_cm[0][1])*100:.1f}%)")
print(f"    FN={int_cm[1][0]:3d}  TP={int_cm[1][1]:3d}  (FN rate: {int_cm[1][0]/(int_cm[1][0]+int_cm[1][1])*100:.1f}%)")

print(f"\n  Test Esterno ({sum(sum(row) for row in ext_cm)} samples):")
print(f"    TN={ext_cm[0][0]:3d}  FP={ext_cm[0][1]:3d}  (FP rate: {ext_cm[0][1]/(ext_cm[0][0]+ext_cm[0][1])*100:.1f}%)")
print(f"    FN={ext_cm[1][0]:3d}  TP={ext_cm[1][1]:3d}  (FN rate: {ext_cm[1][0]/(ext_cm[1][0]+ext_cm[1][1])*100:.1f}%)")

print("\n💡 RACCOMANDAZIONI:")
if abs(f1_drop) > 20:
    print("  🚨 CRITICO: Drop > 20% in F1")
    print("     → Il modello NON generalizza bene su dati esterni")
    print("     → Necessario migliorare il dataset di training")
    print("     → Considerare hard negative mining")

if abs(prec_drop) > 30:
    print("  🚨 CRITICO: Drop > 30% in Precision")
    print("     → Troppi falsi positivi su dati esterni")
    print("     → Aumentare threshold da 0.10 a 0.70")
    print("     → Calibrare il modello")

print("\n" + "="*80)
