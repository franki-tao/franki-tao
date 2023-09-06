package ht_Math.Function;

public abstract class ht_functionN {
    public ht_functionN() {
    }

    public abstract double value(double[] params);

    private double[] arrayCopy(double[] params) {
        double[] ans = new double[params.length];
        System.arraycopy(params, 0, ans, 0, params.length);
        return ans;
    }

    //求偏导数
    public double derivative(double[] params, int index, double fz) {
        double[] tp1 = this.arrayCopy(params);
        double[] tp2 = this.arrayCopy(params);
        tp1[index] += fz;
        tp2[index] -= fz;
        return (this.value(tp1) - this.value(tp2)) / (2 * fz);
    }
}
