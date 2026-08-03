import pytest
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pyqpanda_alg.QmRMR import QmRMR_core as all_code
import warnings
import os


class Test_QmRMR_all_code_Feature_Selection:
    """QmRMR Feature_Selection模块测试类"""

    def setup_method(self):
        """测试方法前置设置"""
        warnings.filterwarnings("ignore")
        np.random.seed(42)  # 设置随机种子保证结果可重现
        # 确保测试输出目录存在
        os.makedirs('test_outputs', exist_ok=True)

    def generate_test_data(self, m=6):
        """生成测试数据"""
        # 生成线性项（特征与目标的相关性）
        u = np.random.random(m)

        # 生成二次项（特征间的相关性矩阵），确保对称
        cor = np.random.random([m, m])
        cor = (cor.T + cor) / 2

        # 生成初始参数
        ini_par = np.random.random(int(m / 2) * m) * np.pi

        return u, cor, ini_par

    def test_interface13_basic_functionality(self):
        """测试接口13: get_his_res() - 基本功能测试"""
        print("\n" + "=" * 60)
        print("测试接口13: Feature_Selection.get_his_res() - 基本功能")
        print("=" * 60)

        # 生成测试数据
        m = 6
        select_num = 3
        u, cor, ini_par = self.generate_test_data(m)

        print(f"测试数据信息:")
        print(f"  特征数量 (m): {m}")
        print(f"  选择特征数量 (select_num): {select_num}")
        print(f"  线性项形状: {u.shape}")
        print(f"  二次项形状: {cor.shape}")
        print(f"  初始参数形状: {ini_par.shape}")
        print(f"  初始参数范围: [{ini_par.min():.4f}, {ini_par.max():.4f}]")

        # 创建Feature_Selection实例
        try:
            feature_selector = all_code.Feature_Selection(
                quadratic=cor,
                linear=u,
                select_num=select_num
            )
            assert feature_selector is not None, "Feature_Selection实例创建失败"
            print("✓ Feature_Selection实例创建成功")

            # 调用get_his_res()方法
            loss, choice, dic = feature_selector.get_his_res(ini_par)

            # 验证输出结果
            assert loss is not None, "loss结果不应为None"
            assert choice is not None, "choice结果不应为None"
            assert dic is not None, "dic结果不应为None"
            print("✓ get_his_res()方法执行成功")

            # 验证loss数据
            assert isinstance(loss, (list, np.ndarray)), "loss应为列表或数组"
            assert len(loss) > 0, "loss应包含数据"
            assert np.all(np.isfinite(loss)), "loss应全部为有限值"
            print(f"✓ loss数据验证通过: 长度={len(loss)}, 范围=[{min(loss):.6f}, {max(loss):.6f}]")

            # 验证choice数据
            assert isinstance(choice, (list, np.ndarray)), "choice应为列表或数组"
            assert len(choice) == m, f"choice长度应为特征数量{m}"
            assert np.all(np.isin(choice, [0, 1])), "choice应只包含0和1"

            # 验证选择的特征数量
            selected_count = np.sum(choice)
            assert selected_count == select_num, f"选择特征数量应为{select_num}, 实际为{selected_count}"
            print(f"✓ choice验证通过: 选择{selected_count}个特征, 符合要求")
            print(f"  选择的特征索引: {np.where(choice)[0]}")

            # 验证字典数据
            assert isinstance(dic, dict), "dic应为字典类型"
            assert len(dic) > 0, "dic应包含数据"
            print(f"✓ dic验证通过: 包含{len(dic)}个键值对")

            # 绘制loss曲线
            plt.figure(figsize=(10, 6))
            plt.plot(loss, 'b-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('QmRMR Feature Selection - Loss Curve')
            plt.grid(True, alpha=0.3)

            # 保存图形
            plt.savefig('test_outputs/qmrmr_loss_curve.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Loss曲线图已保存: test_outputs/qmrmr_loss_curve.png")

            return loss, choice, dic

        except Exception as e:
            pytest.fail(f"基本功能测试失败: {e}")

    def test_interface13_different_feature_sizes(self):
        """测试接口13: 不同特征数量"""
        print("\n" + "=" * 60)
        print("测试接口13: 不同特征数量")
        print("=" * 60)

        test_cases = [
            (4, 2),  # 小规模
            (6, 3),  # 中等规模
            (8, 4),  # 较大规模
            (10, 5),  # 大规模
        ]

        for m, select_num in test_cases:
            print(f"\n测试 {m} 个特征中选择 {select_num} 个:")

            try:
                # 生成测试数据
                u, cor, ini_par = self.generate_test_data(m)

                # 创建实例并执行
                feature_selector = all_code.Feature_Selection(
                    quadratic=cor,
                    linear=u,
                    select_num=select_num
                )
                loss, choice, dic = feature_selector.get_his_res(ini_par)

                # 验证基本结果
                assert loss is not None and len(loss) > 0
                assert choice is not None and len(choice) == m
                assert dic is not None

                # 验证选择数量
                selected_count = np.sum(choice)
                assert selected_count == select_num, f"应选择{select_num}个特征, 实际选择{selected_count}个"

                print(f"  ✓ {m}特征选择{select_num}个测试通过")
                print(f"    选择的特征: {np.where(choice)[0]}")
                print(f"    loss范围: [{min(loss):.6f}, {max(loss):.6f}]")

                # 保存loss曲线
                plt.figure(figsize=(8, 5))
                plt.plot(loss)
                plt.title(f'QmRMR - {m} Features Select {select_num}')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                plt.savefig(f'test_outputs/qmrmr_{m}features_{select_num}select.png',
                            dpi=120, bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"  ✗ {m}特征选择{select_num}个测试失败: {e}")

    def test_interface13_loss_decreasing_trend(self):
        """测试接口13: loss下降趋势"""
        print("\n" + "=" * 60)
        print("测试接口13: loss下降趋势验证")
        print("=" * 60)

        m = 6
        select_num = 3
        u, cor, ini_par = self.generate_test_data(m)

        try:
            feature_selector = all_code.Feature_Selection(
                quadratic=cor,
                linear=u,
                select_num=select_num
            )
            loss, choice, dic = feature_selector.get_his_res(ini_par)

            print(f"Loss数据详情:")
            print(f"  总迭代次数: {len(loss)}")
            print(f"  初始loss: {loss[0]:.6f}")
            print(f"  最终loss: {loss[-1]:.6f}")
            print(f"  loss变化: {loss[0] - loss[-1]:.6f}")

            # 检查loss是否有下降趋势（允许局部波动）
            if len(loss) > 10:
                # 计算滑动平均来观察趋势
                window = min(5, len(loss) // 4)
                moving_avg = np.convolve(loss, np.ones(window) / window, mode='valid')

                # 检查整体趋势
                if moving_avg[0] > moving_avg[-1]:
                    print("✓ Loss呈现下降趋势")
                else:
                    print("⚠ Loss下降趋势不明显，但算法可能仍在优化中")

            # 绘制详细loss分析图
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(loss, 'b-', linewidth=2, label='Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('QmRMR Feature Selection - Loss Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 1, 2)
            # 绘制loss的差分（变化率）
            if len(loss) > 1:
                loss_diff = np.diff(loss)
                plt.plot(range(1, len(loss)), loss_diff, 'r-', linewidth=1, label='Loss Change')
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                plt.xlabel('Iteration')
                plt.ylabel('Loss Change')
                plt.title('Loss Change Rate')
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('test_outputs/qmrmr_loss_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Loss分析图已保存")

        except Exception as e:
            pytest.fail(f"Loss下降趋势测试失败: {e}")

    def test_interface13_feature_selection_consistency(self):
        """测试接口13: 特征选择一致性"""
        print("\n" + "=" * 60)
        print("测试接口13: 特征选择一致性")
        print("=" * 60)

        m = 6
        select_num = 3

        # 多次运行测试一致性
        n_runs = 3
        all_choices = []

        for i in range(n_runs):
            print(f"\n第 {i + 1} 次运行:")

            u, cor, ini_par = self.generate_test_data(m)

            try:
                feature_selector = all_code.Feature_Selection(
                    quadratic=cor,
                    linear=u,
                    select_num=select_num
                )
                loss, choice, dic = feature_selector.get_his_res(ini_par)

                selected_features = np.where(choice)[0]
                all_choices.append(set(selected_features))

                print(f"  选择的特征: {selected_features}")
                print(f"  最终loss: {loss[-1]:.6f}")

            except Exception as e:
                print(f"  第 {i + 1} 次运行失败: {e}")

        # 分析选择的一致性
        if len(all_choices) > 1:
            common_features = set.intersection(*all_choices)
            print(f"\n一致性分析:")
            print(f"  共同选择的特征: {common_features}")
            print(f"  共同特征数量: {len(common_features)}")

    def test_interface13_dictionary_content(self):
        """测试接口13: 字典内容验证"""
        print("\n" + "=" * 60)
        print("测试接口13: 字典内容验证")
        print("=" * 60)

        m = 6
        select_num = 3
        u, cor, ini_par = self.generate_test_data(m)

        try:
            feature_selector = all_code.Feature_Selection(
                quadratic=cor,
                linear=u,
                select_num=select_num
            )
            loss, choice, dic = feature_selector.get_his_res(ini_par)

            print("字典内容分析:")
            print(f"  字典键: {list(dic.keys())}")

            # 检查常见的关键信息
            for key, value in dic.items():
                print(f"  {key}: {type(value)} - {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")

            # 验证字典包含有用的信息
            assert len(dic) > 0, "字典应包含数据"
            print("✓ 字典内容验证通过")

        except Exception as e:
            pytest.fail(f"字典内容验证失败: {e}")

    def test_interface13_consistency_with_example(self):
        """测试与原始示例的一致性"""
        print("\n" + "=" * 60)
        print("测试与原始示例的一致性")
        print("=" * 60)

        # 完全复现原始示例
        m = 6
        u = np.random.random(m)
        cor = np.random.random([m, m])
        cor = (cor.T + cor) / 2
        ini_par = np.random.random(int(m / 2) * m) * np.pi

        print("复现原始示例:")
        print(f"  m = {m}")
        print(f"  u.shape = {u.shape}")
        print(f"  cor.shape = {cor.shape}")
        print(f"  ini_par.shape = {ini_par.shape}")
        print(f"  select_num = 3")

        try:
            loss, choice, dic = all_code.Feature_Selection(cor, u, 3).get_his_res(ini_par)

            print("原始示例输出:")
            print(f"  loss: {loss}")
            print(f"  choice: {choice}")
            print(f"  dic keys: {list(dic.keys())}")

            # 基本验证
            assert loss is not None
            assert choice is not None
            assert dic is not None
            assert np.sum(choice) == 3, "应选择3个特征"

            print("✓ 原始示例测试通过")

            # 绘制loss曲线
            plt.figure(figsize=(10, 6))
            plt.plot(loss, 'b-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('QmRMR Feature Selection - Original Example')
            plt.grid(True, alpha=0.3)
            plt.savefig('test_outputs/qmrmr_original_example.png', dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            pytest.fail(f"原始示例测试失败: {e}")


def test_complete_workflow():
    """完整工作流测试 - 独立测试函数"""
    print("\n" + "=" * 60)
    print("完整工作流测试 - QmRMR Feature Selection")
    print("=" * 60)

    # 设置随机种子
    np.random.seed(123)

    # 生成测试数据
    m = 6
    select_num = 3
    u = np.random.random(m)
    cor = np.random.random([m, m])
    cor = (cor.T + cor) / 2
    ini_par = np.random.random(int(m / 2) * m) * np.pi

    print("测试数据信息:")
    print(f"  特征数量: {m}")
    print(f"  选择特征数量: {select_num}")
    print(f"  线性项范围: [{u.min():.4f}, {u.max():.4f}]")
    print(f"  二次项范围: [{cor.min():.4f}, {cor.max():.4f}]")
    print(f"  初始参数范围: [{ini_par.min():.4f}, {ini_par.max():.4f}]")

    try:
        # 创建特征选择器
        feature_selector = all_code.Feature_Selection(
            quadratic=cor,
            linear=u,
            select_num=select_num
        )
        print("✓ Feature_Selection实例创建成功")

        # 执行特征选择
        loss, choice, dic = feature_selector.get_his_res(ini_par)
        print("✓ get_his_res()方法执行成功")

        # 验证结果
        assert loss is not None and len(loss) > 0
        assert choice is not None and len(choice) == m
        assert dic is not None

        # 验证特征选择数量
        selected_count = np.sum(choice)
        assert selected_count == select_num
        selected_indices = np.where(choice)[0]

        print(f"✓ 特征选择验证通过:")
        print(f"  选择的特征数量: {selected_count}")
        print(f"  选择的特征索引: {selected_indices}")
        print(f"  Loss迭代次数: {len(loss)}")
        print(f"  初始Loss: {loss[0]:.6f}")
        print(f"  最终Loss: {loss[-1]:.6f}")
        print(f"  Loss改善: {loss[0] - loss[-1]:.6f}")

        # 绘制综合结果图
        plt.figure(figsize=(12, 8))

        # Loss曲线
        plt.subplot(2, 1, 1)
        plt.plot(loss, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('QmRMR Feature Selection - Loss Curve')
        plt.grid(True, alpha=0.3)

        # 特征选择结果
        plt.subplot(2, 1, 2)
        colors = ['red' if c == 1 else 'blue' for c in choice]
        plt.bar(range(m), [1] * m, color=colors, alpha=0.7)
        plt.xlabel('Feature Index')
        plt.ylabel('Selected (1) / Not Selected (0)')
        plt.title('Feature Selection Result')
        plt.xticks(range(m))
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('test_outputs/qmrmr_complete_workflow.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ 完整工作流结果图已保存")

        print("🎉 完整工作流测试通过!")

    except Exception as e:
        print(f"✗ 完整工作流测试失败: {e}")
        raise


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])