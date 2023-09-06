package ht_Math.Solver.optimizer;

import ht_Math.Function.ht_functionN;
import ht_Math.Matrix.h_matrix;
import ht_Math.Solver.domain.problem;

import java.util.ArrayList;
import java.util.Arrays;

//Limited memory BFGS
public class L_BFGS extends optimize {
    public double[] X0;
    public double[] result;

    public L_BFGS(problem prob, int iterations, double error, double[] X0) {
        super(prob, iterations, error);
        this.X0 = X0;
    }

    private double[] Hp(double[][] HO, double[] p, ArrayList<double[]> sks, double[] xk, ArrayList<double[]> yks) {
        int m_t = sks.size();
        double[] q = prob.calGrad(xk);
        double[] a = new double[m_t];
        double[] b = new double[m_t];
        double[] s, y;
        double rho_i;
        for (int i = m_t - 1; i > -1; i--) {
            s = sks.get(i);
            y = yks.get(i);
            rho_i = 1 / utilities.dot(y, s);
            a[i] = rho_i * utilities.dot(s, q);
            q = utilities.add(1, q, -a[i], y);
        }
        double[] r = utilities.dot(HO, q);
        for (int i = 0; i < m_t; i++) {
            s = sks.get(i);
            y = yks.get(i);
            rho_i = 1 / utilities.dot(y, s);
            b[i] = rho_i * utilities.dot(y, r);
            r = utilities.add(1, r, (a[i] - b[i]), s);
        }
        return r;
    }

    public boolean optim() {
        double[] xk = utilities.arrayCopy(X0);
        double c2 = 0.9;
        int m = 10;
        double[][] I = utilities.identity(xk.length);
        double[][] Hk = utilities.identity(xk.length);
        ArrayList<double[]> sks = new ArrayList<>();
        ArrayList<double[]> yks = new ArrayList<>();


        double[] gk, pk, xk1, gk1, sk, yk;
        double alpha, rho_k;
        for (int i = 0; i < iterations; i++) {
            gk = prob.calGrad(xk);
            pk = utilities.minus(Hp(I, gk, sks, xk, yks));
            alpha = utilities.step_length(prob, xk, 1, pk, c2);
            xk1 = utilities.add(1, xk, alpha, pk);
            gk1 = prob.calGrad(xk1);
            sk = utilities.add(1, xk1, -1, xk);
            yk = utilities.add(1, gk1, -1, gk);

            sks.add(sk);
            yks.add(yk);

            if (sks.size() > m) {
                sks.remove(0);
                yks.remove(0);
            }
            if (utilities.norm(xk1, xk) < error) {
                result = utilities.arrayCopy(xk);
                return true;
            }
            xk = utilities.arrayCopy(xk1);
        }
        return false;
    }

    public static void main(String[] args) {
        double[] x0 = {0, 0};
        problem prob = new problem(new ht_functionN() {
            @Override
            public double value(double[] params) {
                return 100 * (params[1] - params[0] * params[0]) * (params[1] - params[0] * params[0]) + (1 - params[0]) * (1 - params[0]);
            }
        }) {

            @Override
            public double[] calGrad(double[] x) {
                return new double[]{200 * (x[1] - x[0] * x[0]) * (-2 * x[0]) + 2 * (x[0] - 1), 200 * (x[1] - x[0] * x[0])};
            }

            @Override
            public h_matrix calHess(double[] x) {
                return null;
            }
        };
        L_BFGS l_bfgs = new L_BFGS(prob, 1000, 1e-5, x0);
        if (l_bfgs.optim()) {
            System.out.println(Arrays.toString(l_bfgs.result));
        }
    }
}
