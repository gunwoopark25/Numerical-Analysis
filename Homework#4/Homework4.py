# -*- coding: utf-8 -*-
"""
Spalding's Law of the Wall (1961) : u+ vs y+ 곡선
==================================================
y+ = u+ + exp(-kB)*[exp(k u+) - 1 - (k u+) - (k u+)^2/2 - (k u+)^3/6]

각 y+ 값에 대해 u+ 를 구하려면 위 음함수를 root-finding 으로 풀어야 한다.
본 스크립트는 동일한 y+ 그리드에 대해 Bisection 과 Newton 으로 각각 u+ 를 풀고,
(1) 두 방법으로 얻은 u+ vs y+ 곡선을 점성저층/로그법칙 점근선과 함께 표시,
(2) 세 대표 y+ 위치(viscous sublayer / buffer / log layer)에서의 수렴과정을 도시.

실행 방법 (CLI)
---------------
    Windows :  python spalding_wall.py
               python -u spalding_wall.py            # unbuffered
               python spalding_wall.py > log.txt     # CLI 출력 파일 저장
    Linux/macOS :  python3 spalding_wall.py
                   python3 spalding_wall.py | tee log.txt

필요 패키지
-----------
    pip install numpy matplotlib
"""
from pathlib import Path
import platform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ===== 한글 폰트 자동 설정 =====
_os = platform.system()
if _os == 'Windows':
    mpl.rcParams['font.family'] = 'Malgun Gothic'
elif _os == 'Darwin':
    mpl.rcParams['font.family'] = 'AppleGothic'
else:
    try:
        from matplotlib import font_manager as fm
        for cand in ['NanumGothic', 'Noto Sans CJK KR', 'UnDotum']:
            if any(cand in f.name for f in fm.fontManager.ttflist):
                mpl.rcParams['font.family'] = cand
                break
    except Exception:
        pass
mpl.rcParams['axes.unicode_minus']   = False
mpl.rcParams['mathtext.fontset']     = 'dejavuserif'
mpl.rcParams['figure.dpi']           = 100

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path.cwd()

# ===== Spalding 상수 =====
KAPPA  = 0.41             # von Kármán 상수
B      = 5.0              # 적층 상수
EXP_NKB = np.exp(-KAPPA * B)


# ===== Spalding 식 / 잔차 / 도함수 =====
def spalding_yplus(uplus):
    """y+ = F(u+).  순방향 평가 (참조 / 검증용)."""
    ku = KAPPA * uplus
    return uplus + EXP_NKB * (np.exp(ku) - 1.0 - ku - ku**2 / 2.0 - ku**3 / 6.0)


def f(uplus, yplus):
    """잔차  f(u+; y+) = F(u+) - y+ ."""
    ku = KAPPA * uplus
    return uplus + EXP_NKB * (np.exp(ku) - 1.0 - ku - ku**2 / 2.0 - ku**3 / 6.0) - yplus


def df(uplus):
    """df/du+ = 1 + kappa * exp(-kB) * [exp(k u+) - 1 - k u+ - (k u+)^2/2]."""
    ku = KAPPA * uplus
    return 1.0 + KAPPA * EXP_NKB * (np.exp(ku) - 1.0 - ku - ku**2 / 2.0)


# ===== Bisection =====
def bisection(yplus, a=0.0, b=50.0, tol=1e-10, max_iter=80, record=False):
    """단조 증가 함수이므로 항상 안전하게 수렴.  bracket [0, 50] 으로 충분."""
    fa, fb = f(a, yplus), f(b, yplus)
    # 매우 큰 y+ 에 대비해 상한 자동 확장
    while fb < 0 and b < 1e6:
        b *= 2.0
        fb = f(b, yplus)
    hist = [] if record else None
    c    = 0.5 * (a + b)
    for n in range(1, max_iter + 1):
        c   = 0.5 * (a + b)
        fc  = f(c, yplus)
        err = 0.5 * (b - a)
        if record:
            hist.append({'n': n, 'a': a, 'b': b, 'c': c, 'fc': fc, 'err': err})
        if abs(fc) < tol or err < tol:
            break
        if f(a, yplus) * fc < 0:
            b = c
        else:
            a = c
    return (c, n, hist) if record else (c, n)


# ===== Newton-Raphson =====
def _smart_initial(yplus):
    """선형 sublayer / 로그법칙 점근에 기반한 초기 추정값."""
    if yplus < 5.0:
        return yplus                               # viscous sublayer : u+ ≈ y+
    return (1.0 / KAPPA) * np.log(max(yplus, 1.1)) + B  # log law

def newton(yplus, x0=None, tol=1e-12, max_iter=40, record=False):
    """지수 항 때문에 초기값이 너무 멀면 발산.  smart initial 사용 + step damping 안전장치."""
    if x0 is None:
        x0 = _smart_initial(yplus)
    x        = float(x0)
    fx, dfx  = f(x, yplus), df(x)
    hist     = ([{'n': 0, 'x': x, 'fx': fx, 'dfx': dfx, 'dx': None}]
                if record else None)
    n_done   = 0
    for n in range(1, max_iter + 1):
        if abs(dfx) < 1e-15:
            break
        step  = fx / dfx
        # damping : 한 step 이 5 보다 크면 절반으로 (지수 발산 방지)
        if abs(step) > 5.0:
            step *= 5.0 / abs(step)
        x_new = x - step
        dx    = abs(x_new - x)
        x     = x_new
        fx, dfx = f(x, yplus), df(x)
        if record:
            hist.append({'n': n, 'x': x, 'fx': fx, 'dfx': dfx, 'dx': dx})
        n_done = n
        if dx < tol or abs(fx) < tol:
            break
    return (x, n_done, hist) if record else (x, n_done)


# ===== 1) u+ vs y+ 마스터 곡선 =====
def compute_curve():
    yp = np.logspace(-1, 3.5, 80)        # y+ ∈ [0.1, ~3162]
    u_b = np.empty_like(yp)
    u_n = np.empty_like(yp)
    n_b = np.empty_like(yp, dtype=int)
    n_n = np.empty_like(yp, dtype=int)
    for i, y in enumerate(yp):
        u_b[i], n_b[i] = bisection(y)
        u_n[i], n_n[i] = newton(y)
    return yp, u_b, u_n, n_b, n_n


def plot_master_curve(yp, u_b, u_n, n_b, n_n, save_path):
    fig = plt.figure(figsize=(13, 8))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1.7, 1], hspace=0.35, wspace=0.28)
    ax_main = fig.add_subplot(gs[0, :])
    ax_diff = fig.add_subplot(gs[1, 0])
    ax_iter = fig.add_subplot(gs[1, 1])

    # --- (1) Master curve ---
    yp_ref = np.logspace(-1, 3.5, 400)
    ax_main.plot(yp_ref, yp_ref, 'k--', lw=1.0, alpha=0.6,
                 label=r'Linear sublayer  $u^+=y^+$')
    log_law = (1.0 / KAPPA) * np.log(np.clip(yp_ref, 1e-12, None)) + B
    mask    = yp_ref >= 11   # 의미 있는 구간만
    ax_main.plot(yp_ref[mask], log_law[mask], 'k:', lw=1.0, alpha=0.6,
                 label=r'Log law  $u^+=\frac{1}{\kappa}\ln y^+ + B$')

    ax_main.plot(yp, u_b, '-',  color='tab:blue',   lw=2.0,
                 label='Spalding (Bisection)')
    ax_main.plot(yp, u_n, 'o',  color='tab:orange', ms=4.5, mfc='none', mew=1.3,
                 label='Spalding (Newton)')

    # 영역 구분
    ax_main.axvspan(0.1, 5,   color='lightyellow', alpha=0.35, zorder=0)
    ax_main.axvspan(5,   30,  color='lightgreen',  alpha=0.20, zorder=0)
    ax_main.axvspan(30,  3500, color='lightblue',  alpha=0.20, zorder=0)
    ymax = u_b.max() * 1.05
    ax_main.text(1.0,  ymax * 0.92, 'viscous\nsublayer', ha='center',
                 fontsize=9, color='#666')
    ax_main.text(12.5, ymax * 0.92, 'buffer\nlayer',     ha='center',
                 fontsize=9, color='#666')
    ax_main.text(300,  ymax * 0.92, 'log layer',         ha='center',
                 fontsize=9, color='#666')

    ax_main.set_xscale('log')
    ax_main.set_xlim(0.1, 3500)
    ax_main.set_ylim(0, ymax)
    ax_main.set_xlabel(r'$y^+$')
    ax_main.set_ylabel(r'$u^+$')
    ax_main.set_title("Spalding's Law of the Wall : $u^+$ vs $y^+$")
    ax_main.grid(True, which='both', alpha=0.3)
    ax_main.legend(loc='upper left', fontsize=10)

    # --- (2) 두 방법 사이 차이 ---
    diff = np.abs(u_b - u_n)
    ax_diff.semilogx(yp, np.maximum(diff, 1e-16), 'o-', ms=3, color='tab:red')
    ax_diff.set_xlabel(r'$y^+$')
    ax_diff.set_ylabel(r'$|u^+_{\,bisec} - u^+_{\,newton}|$')
    ax_diff.set_yscale('log')
    ax_diff.set_title('두 방법 결과 차이 (교차 검증)')
    ax_diff.grid(True, which='both', alpha=0.3)

    # --- (3) iteration 수 비교 ---
    ax_iter.semilogx(yp, n_b, 's-', ms=4, color='tab:blue',   label='Bisection')
    ax_iter.semilogx(yp, n_n, 'o-', ms=4, color='tab:orange', label='Newton')
    ax_iter.set_xlabel(r'$y^+$')
    ax_iter.set_ylabel('iteration 횟수')
    ax_iter.set_title('수렴까지의 iteration 횟수')
    ax_iter.grid(True, which='both', alpha=0.3)
    ax_iter.legend()

    fig.suptitle("Spalding's Law of the Wall : Bisection vs Newton-Raphson",
                 fontsize=13, fontweight='bold')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"  [저장] {save_path}")
    plt.show()


# ===== 2) 세 대표 y+ 에서의 수렴 상세 =====
def plot_convergence_detail(save_path):
    targets = [
        (1.0,   'viscous sublayer',  'tab:green'),
        (30.0,  'buffer layer',      'tab:orange'),
        (300.0, 'log layer',         'tab:purple'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    fig.suptitle('대표 $y^+$ 위치에서의 수렴 거동',
                 fontsize=13, fontweight='bold')

    for col, (yp, label, color) in enumerate(targets):
        # 각 방법으로 풀기 (record=True)
        u_b, n_b, hist_b = bisection(yp, record=True)
        u_n, n_n, hist_n = newton  (yp, record=True)

        # 콘솔 출력
        print(f"\n  ----- y+ = {yp:g}  ({label}) -----")
        print(f"    Bisection : u+ = {u_b:.10f}  ({n_b} iter)")
        print(f"    Newton    : u+ = {u_n:.10f}  ({n_n} iter)")
        print(f"    diff      : {abs(u_b - u_n):.3e}")

        # ── 상단 : 함수 + iteration 위치 ──
        ax = axes[0, col]
        u_axis = np.linspace(0, max(u_b, u_n) * 1.4 + 1, 400)
        ax.plot(u_axis, [f(u, yp) for u in u_axis], 'k-', lw=1.2,
                label=r'$f(u^+) = F(u^+) - y^+$')
        ax.axhline(0, color='gray', lw=0.6)
        ax.axvline(u_b, color='crimson', ls='--', lw=0.9, alpha=0.7,
                   label=fr'root $u^*={u_b:.4f}$')

        # bisection 수렴점들 (마지막 6개만 표시)
        for r in hist_b[-6:]:
            ax.plot(r['c'], r['fc'], 's', color='tab:blue', ms=5,
                    mfc='none', mew=1.2, alpha=0.85)
        # newton 반복해 표시
        for i, r in enumerate(hist_n):
            ax.plot(r['x'], r['fx'], 'o', color='tab:orange', ms=6,
                    mfc='white', mew=1.3, alpha=0.9)
            if i < 4:
                ax.annotate(f'$x_{{{r["n"]}}}$', (r['x'], r['fx']),
                            textcoords='offset points', xytext=(5, 6),
                            fontsize=8, color='tab:orange')

        ax.set_title(fr'$y^+={yp:g}$  ({label})')
        ax.set_xlabel(r'$u^+$')
        ax.set_ylabel(r'$f(u^+)$')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

        # ── 하단 : |f| 잔차 수렴 ──
        ax2 = axes[1, col]
        ns_b = [r['n']  for r in hist_b]
        rs_b = [max(abs(r['fc']), 1e-18) for r in hist_b]
        ns_n = [r['n']  for r in hist_n]
        rs_n = [max(abs(r['fx']), 1e-18) for r in hist_n]
        ax2.semilogy(ns_b, rs_b, 's-', color='tab:blue',
                     label=f'Bisection ({n_b} iter)', ms=4)
        ax2.semilogy(ns_n, rs_n, 'o-', color='tab:orange',
                     label=f'Newton ({n_n} iter)',    ms=5)
        ax2.set_xlabel('iteration n')
        ax2.set_ylabel(r'$|f|$')
        ax2.set_title('잔차 수렴')
        ax2.grid(True, which='both', alpha=0.3)
        ax2.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\n  [저장] {save_path}")
    plt.show()


# ===== Main =====
if __name__ == '__main__':
    print("=" * 70)
    print(" Spalding's Law of the Wall : u+ vs y+ 풀이")
    print(f"  kappa = {KAPPA}, B = {B}, exp(-kB) = {EXP_NKB:.6e}")
    print("=" * 70)

    # 1) 마스터 곡선
    print("\n[1] u+ vs y+ 마스터 곡선 계산 (80 points, log-spaced 0.1 ~ 3162)...")
    yp, u_b, u_n, n_b, n_n = compute_curve()
    print(f"  Bisection iter : min={n_b.min():2d}  max={n_b.max():2d}  "
          f"mean={n_b.mean():4.1f}")
    print(f"  Newton    iter : min={n_n.min():2d}  max={n_n.max():2d}  "
          f"mean={n_n.mean():4.1f}")
    print(f"  최대 차이      : max|u_b - u_n| = {np.max(np.abs(u_b - u_n)):.3e}")

    # 대표 검증
    print("\n[검증] 주요 y+ 에서의 u+ 값 :")
    print(f"  {'y+':>8} {'u+ (Bisec)':>14} {'u+ (Newton)':>14} "
          f"{'log law':>12} {'linear':>10}")
    for yt in [0.5, 1, 5, 11, 30, 100, 300, 1000]:
        ub_t, _ = bisection(yt)
        un_t, _ = newton(yt)
        log_t   = (1.0 / KAPPA) * np.log(yt) + B if yt > 1 else float('nan')
        print(f"  {yt:>8.1f} {ub_t:>14.8f} {un_t:>14.8f} "
              f"{log_t:>12.4f} {yt:>10.4f}")

    plot_master_curve(yp, u_b, u_n, n_b, n_n,
                      HERE / 'spalding_master.png')

    # 2) 수렴 상세
    print("\n[2] 대표 y+ 위치에서의 수렴 과정 상세 ...")
    plot_convergence_detail(HERE / 'spalding_convergence.png')

    print("\n" + "=" * 70)
    print(" 완료")
    print("=" * 70)
