import numpy as np
from scipy.spatial import ConvexHull

def compute_card(W_complex):
    points = np.column_stack((W_complex.real, W_complex.imag))

    hull = ConvexHull(points)

    # 1. c_W (Perimeter / 2pi)
    perimeter = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        perimeter += np.linalg.norm(p1 - p2)
    c_w = perimeter / (2 * np.pi)

    # Grid of angles for support function
    K = 3600
    thetas = np.linspace(0, 2*np.pi, K, endpoint=False)

    R_in = np.inf
    rho = 0.0

    xi_list = []
    g_hat_data = []

    for th in thetas:
        # Support quantizer: w*(theta) = argmax_{w in W} Re(e^{-j theta} w)
        proj = np.real(np.exp(-1j * th) * W_complex)
        idx_opt = np.argmax(proj)
        w_opt = W_complex[idx_opt]

        h_theta = proj[idx_opt]
        if h_theta < R_in:
            R_in = h_theta

        dist_to_circle = np.min(np.abs(np.exp(1j * th) - W_complex))
        if dist_to_circle > rho:
            rho = dist_to_circle

        xi = w_opt * np.exp(-1j * th)
        xi_list.append(xi)
        g_hat_data.append(w_opt)

    R_in = max(0, R_in)

    xi_arr = np.array(xi_list)
    E_xi = np.mean(xi_arr)
    sigma2 = np.mean(np.abs(xi_arr - E_xi)**2)

    ghat = np.fft.fft(g_hat_data) / K
    lambda_param = abs(ghat[1] / (ghat[0] + 1e-12)) # simple approx or can be adjusted depending on k*

    return {
        "c_W": c_w,
        "L_W": 20 * np.log10(c_w + 1e-12),
        "R_in": R_in,
        "rho": rho,
        "sigma2": sigma2,
        "lambda": lambda_param
    }

if __name__ == "__main__":
    from beamformer.element_model import MeasuredVaractor
    el = MeasuredVaractor()
    print("Canonical Measured Varactor:")
    c = compute_card(el.c_grid)
    for k, v in c.items():
        print(f"  {k}: {v}")
