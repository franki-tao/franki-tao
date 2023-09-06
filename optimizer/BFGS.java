import Function.ht_functionN;
import domain.problem;

import java.util.Arrays;

//Broyden-Fletcher-Goldfarb-Shanno算法 （拟牛顿法）
public class BFGS extends optimize{
    public double[] X0;
    public double[] result;
    public BFGS(problem prob, int iterations, double error, double[] X0) {
        super(prob, iterations, error);
        this.X0 = X0;
    }

    public boolean optim(){
        double[] xk = utilities.arrayCopy(X0);
        double c2 = 0.9;
        double[][] I = utilities.identity(xk.length);
        double[][] Hk = utilities.identity(xk.length);
        double[] gk, pk, xk1, gk1, sk, yk;
        double alpha, rho_k;
        double[][] Hk1, tp1, tp2;
        for (int i = 0; i < iterations; i++) {
            gk = prob.calGrad(xk);
            pk = utilities.minus(utilities.dot(Hk, gk));
            alpha = utilities.step_length(prob, xk, 1, pk, c2);
            xk1 = utilities.add(1,xk,alpha,pk);
            gk1 = prob.calGrad(xk1);
            sk = utilities.add(1,xk1, -1, xk);
            yk = utilities.add(1,gk1, -1, gk);
            rho_k = 1/utilities.dot(yk,sk);
            tp1 = utilities.dot(utilities.add(1, I, -rho_k,utilities.outer(sk,yk)), Hk);
            tp2 = utilities.add(1, I, -rho_k,utilities.outer(yk,sk));
            Hk1 = utilities.add(1, utilities.dot(tp1,tp2), rho_k, utilities.outer(sk,sk));
            if(utilities.norm(xk1,xk)<error) {
                result = utilities.arrayCopy(xk);
                return true;
            }
            Hk = utilities.matrixCopy(Hk1);
            xk = utilities.arrayCopy(xk1);
        }
        return false;
    }

    public static void main(String[] args) {
        double[] x0 = {0,0};
        problem prob = new problem(new ht_functionN() {
            @Override
            public double value(double[] params) {
                return 100*(params[1]-params[0]*params[0])*(params[1]-params[0]*params[0]) + (1-params[0])*(1-params[0]);
            }
        }) {

            @Override
            public double[] calGrad(double[] x) {
                return new double[]{200*(x[1]-x[0]*x[0])*(-2*x[0])+2*(x[0]-1), 200*(x[1]-x[0]*x[0])};
            }

        };

        BFGS bfgs = new BFGS(prob, 1000, 1e-5, x0);
        if(bfgs.optim()){
            System.out.println(Arrays.toString(bfgs.result));
        }
    }
}
