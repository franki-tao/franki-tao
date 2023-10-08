import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.StatUtils;

import java.util.Arrays;
import java.util.Comparator;

//主成分分析法
public class PrincipalComponentAnalysis {
    /**
    说明：PCA(Principal Component Analysis)，即主成分分析方法，是一种使用最广泛的数据降维算法。
    PCA的主要思想是将n维特征映射到k维上，这k维是全新的正交特征也被称为主成分，是在原有n维特征的基础上
    重新构造出来的k维特征。PCA的工作就是从原始的空间中顺序地找一组相互正交的坐标轴，新的坐标轴的选择
    与数据本身是密切相关的。其中，第一个新坐标轴选择是原始数据中方差最大的方向，第二个新坐标轴选取是与
    第一个坐标轴正交的平面中使得方差最大的，第三个轴是与第1,2个轴正交的平面中方差最大的。依次类推，
    可以得到n个这样的坐标轴。通过这种方式获得的新的坐标轴，我们发现，大部分方差都包含在前面k个坐标
    轴中，后面的坐标轴所含的方差几乎为0。于是，我们可以忽略余下的坐标轴，只保留前面k个含有绝大部分
    方差的坐标轴。事实上，这相当于只保留包含绝大部分方差的维度特征，而忽略包含方差几乎为0的特征维度，
    实现对数据特征的降维处理。

    一般步骤 (将n维数据，降到k维)
    输入 X = {x_1, x_2, ... , x_n}
    1) 去平均值
    2) 计算协方差矩阵 /frac{1}{n}X^{T}X
    3) 计算协方差矩阵的特征值与特征向量 (采用特征值分解法)
    4) 对特征值从大到小排序，选择其中最大的k个。然后将其对应的k个特征向量分别作为行向量组成特征向量矩阵P
    5) 将数据转换到k个特征向量构建的新空间中，即Y=PX

    实现：huang tao
     */
    public RealMatrix principleData;
    public int k;
    public RealMatrix resultData;

    public PrincipalComponentAnalysis() {
    }

    public PrincipalComponentAnalysis(RealMatrix principleData, int k) {
        this.principleData = principleData;
        this.k = k;
    }

    /*
        归一化数据
     */
    private RealMatrix Normalization(RealMatrix X) {
        double temp;
        double[] column;
        RealMatrix res = X.copy();
        for (int i = 0; i < X.getColumnDimension(); i++) {
            column = X.getColumn(i);
            temp = StatUtils.mean(column);
            RealVector v = new ArrayRealVector(column);
            v = v.mapSubtract(temp);
            res.setColumnVector(i, v);
        }
        return res;
    }

    /*
        计算协方差矩阵
     */
    private RealMatrix calCorrMatrix(RealMatrix X) {
        return X.transpose().multiply(X);//.scalarMultiply(1d / X.getColumnDimension());
    }

    private Integer[] maxIndex(double[] arr, int k) {
        // 创建包含索引的数组，并排序
        Integer[] indexArray = new Integer[arr.length];
        for (int i = 0; i < arr.length; i++) {
            indexArray[i] = i;
        }

        // 使用比较器按照数组的值进行排序
        Arrays.sort(indexArray, Comparator.comparingDouble(index -> arr[index]));

        // 选择排序后的前k个索引
        return Arrays.copyOfRange(indexArray, arr.length - k, arr.length);
    }

    /*
        特征值分解
     */
    private RealMatrix EigenvalueDecomposition(RealMatrix X, int k) {
        EigenDecomposition eigenDecomposition = new EigenDecomposition(X);
        RealMatrix d = eigenDecomposition.getD();
        RealMatrix v = eigenDecomposition.getV();
        double[] arr = new double[d.getColumnDimension()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = d.getEntry(i, i);
        }
        Integer[] index = maxIndex(arr, k);
        double[][] mat = new double[k][X.getRowDimension()];
        for (int i = 0; i < k; i++) {
            mat[i] = v.getColumn(index[i]);
        }
        RealMatrix matrix = MatrixUtils.createRealMatrix(mat);
        return matrix.transpose();
    }

    public void PCA() {
        //归一化
        RealMatrix X = this.Normalization(principleData);
        //计算协方差矩阵
        RealMatrix corrMatrix = this.calCorrMatrix(X);
        //特征值分解
        RealMatrix D = this.EigenvalueDecomposition(corrMatrix, k);
        //降维
        resultData = X.multiply(D);
    }

    public static void main(String[] args) {
        double[][] data = {{-1, 1}, {-2, -1}, {-3, -2}, {1, 1}, {2, 1}, {3, 2}};
        int k = 1; // 要找到最大的3个元素的索引
        PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis(MatrixUtils.createRealMatrix(data), k);
        pca.PCA();
        System.out.println(pca.resultData);

    }
}
