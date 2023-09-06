package ht_Math.Solver.optimizer;

import ht_Math.Function.ht_functionN;
import ht_Math.Matrix.h_matrix;
import ht_Math.Solver.domain.problem;

import java.util.Arrays;

//共轭梯度法
public class conjugate_gradient extends optimize{
    public double[] X0;
    public double[] result;
    public conjugate_gradient(double[] X0, problem prob, int iterations, double error) {
        super(prob, iterations, error);
        this.X0 = X0;
    }

    public boolean optim(){
        double[] xk = utilities.arrayCopy(X0);
        double c2 = 0.1;
        double fk = prob.f.value(xk);
        double[] gk = prob.calGrad(xk);

        double[] pk = utilities.minus(gk);

        double alpha, beta_k1;
        double[] xk1, gk1, pk1;
        for (int i = 0; i < this.iterations; i++) {
            alpha = utilities.step_length(prob, xk, 1, pk, c2);
            xk1 = utilities.add(1,xk,alpha,pk);
            gk1 = prob.calGrad(xk1);
            beta_k1 = utilities.dot(gk1,gk1)/utilities.dot(gk, gk);
            pk1 = utilities.add(-1, gk1, beta_k1, pk);

            if(utilities.norm(xk1, xk)<error) {
                result = utilities.arrayCopy(xk1);
                return true;
            }

            xk = utilities.arrayCopy(xk1);
            gk = utilities.arrayCopy(gk1);
            pk = utilities.arrayCopy(pk1);
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

            @Override
            public h_matrix calHess(double[] x) {
                return null;
            }
        };
        conjugate_gradient optimize = new conjugate_gradient(x0, prob, 1000, 1e-5);
        if(optimize.optim()) {
            System.out.println(Arrays.toString(optimize.result));
        }
    }
}
