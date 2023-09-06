import domain.problem;

public class utilities {
    public static double[] arrayCopy(double[] a) {
        double[] res = new double[a.length];
        System.arraycopy(a, 0, res, 0, res.length);
        return res;
    }
    public static double[][] matrixCopy(double[][] m) {
        double[][] ans = new double[m.length][m[0].length];
        for (int i = 0; i < ans.length; i++) {
            System.arraycopy(m[i], 0, ans[i], 0, ans[0].length);
        }
        return ans;
    }

    public static double[] add(double a, double[] arr1, double b, double[] arr2) {
        double[] res = new double[arr1.length];
        for (int i = 0; i < res.length; i++) {
            res[i] = a * arr1[i] + b * arr2[i];
        }
        return res;
    }

    public static double[][] outer(double[] a, double[] b) {
        double[][] ans = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                ans[i][j] = a[i]*b[j];
            }
        }
        return ans;
    }

    public static double[][] dot(double a, double[][] b) {
        double[][] ans = new double[b.length][b[0].length];
        for (int i = 0; i < b.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                ans[i][j] = a*b[i][j];
            }
        }
        return ans;
    }

    public static double[][] dot(double[][] a, double[][] b) {
        double[][] ans = new double[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                for (int k = 0; k < a[0].length; k++) {
                    ans[i][j] += (a[i][k]*b[k][j]);
                }
            }
        }
        return ans;
    }

    public static double[][] add(double a, double b, double[][] m) {
        double[][] bm = dot(b, m);
        for (int i = 0; i < bm.length; i++) {
            for (int j = 0; j < bm[0].length; j++) {
                bm[i][j] += a;
            }
        }
        return bm;
    }

    public static double[][] add(double a, double[][] m1, double b, double[][] m2) {
        double[][] am = dot(a, m1);
        double[][] bm = dot(b, m2);
        for (int i = 0; i < bm.length; i++) {
            for (int j = 0; j < bm[0].length; j++) {
                bm[i][j] += am[i][j];
            }
        }
        return bm;
    }

    public static double dot(double[] a, double[] b) {
        double res = 0;
        for (int i = 0; i < a.length; i++) {
            res += (a[i] * b[i]);
        }
        return res;
    }

    public static double[] dot(double[][] a, double[] b) {
        double[] ans = new double[a.length];
        for (int i = 0; i < ans.length; i++) {
            ans[i] = dot(a[i], b);
        }
        return ans;
    }

    public static boolean wolfe(problem prob, double[] xk, double alpha, double[] pk) {
        double c1 = 1e-4;
        return prob.f.value(add(1, xk, alpha, pk)) <= prob.f.value(xk) + c1 * alpha * dot(prob.calGrad(xk), pk);
    }

    public static boolean strong_wolfe(problem prob, double[] xk, double alpha, double[] pk, double c2) {
        return wolfe(prob, xk, alpha, pk) && Math.abs(dot(prob.calGrad(add(1, xk, alpha, pk)), pk)) <= c2 *
                Math.abs(dot(prob.calGrad(xk), pk));
    }

    public static boolean gold_stein(problem prob, double[] xk, double alpha, double[] pk, double c) {
        return (prob.f.value(xk) + (1 - c) * alpha * dot(prob.calGrad(xk), pk) <= prob.f.value(add(1, xk, alpha, pk))) &&
                (prob.f.value(add(1, xk, alpha, pk)) <= prob.f.value(xk) + c * alpha * dot(prob.calGrad(xk), pk));
    }

    public static double[] minus(double[] x) {
        double[] ans = new double[x.length];
        for (int i = 0; i < ans.length; i++) {
            ans[i] = -x[i];
        }
        return ans;
    }

    public static double norm(double[] arr1, double[] arr2) {
        double ans = 0d;
        for (int i = 0; i < arr1.length; i++) {
            ans += (arr1[i]-arr2[i])*(arr1[i]-arr2[i]);
        }
        return Math.sqrt(ans);
    }

    public static double[][] identity(int l) {
        double[][] ans = new double[l][l];
        for (int i = 0; i < l; i++) {
            ans[i][i] = 1;
        }
        return ans;
    }

    public static double step_length(problem prob, double[] xk, double alpha, double[] pk, double c2) {
        double l=0, h=1;
        double half;
        for (int i = 0; i < 20; i++) {
            if(strong_wolfe(prob, xk, alpha, pk, c2)) {
                return alpha;
            }
            half = (l+h)/2;
            alpha = -dot(prob.calGrad(add(1, xk, l, pk)), pk)*h*h/(2*(prob.f.value(add(1,xk,h,pk)) -
                    prob.f.value(add(1,xk,l,pk)) - dot(prob.calGrad(add(1, xk, l, pk)), pk)*h));

            if(alpha<l || alpha > h) {
                alpha = half;
            }
            if(dot(prob.calGrad(add(1, xk, alpha, pk)), pk) > 0) {
                h = alpha;
            }
            else {
                l = alpha;
            }
        }
        return alpha;
    }
}
