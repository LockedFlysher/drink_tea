# Z1 Whole-body MPC Demo (`z1_whole_body_mpc_demo.py`)

本脚本在平面浮动基座 + 6 自由度机械臂模型上，演示一个基于速度层的 Whole-body MPC 控制器：

- 状态：`x ∈ R⁹ = [x_base, y_base, φ_base, q1..q6]`
- 控制：`u ∈ R⁹ = [v_x, v_y, φ̇_base, q̇1..q̇6]`
- 动力学：`x_{k+1} = x_k + dt * u_k`
- 成本：末端位置 + 姿态跟踪（Pinocchio FK）+ 关节约束 + 控制能量

可视化与仿真使用 MuJoCo 模型 `robot_description/z1_floating_base.xml`，符号运算和 MPC 使用 CasADi + Pinocchio（`pinocchio.casadi`）。

---

## 1. 目录结构与运行位置

在项目根目录下有：

- `z1_whole_body_mpc_demo.py`：本 Demo 脚本
- `robot_description/z1.urdf`：Z1 机械臂 URDF（供 Pinocchio 使用）
- `robot_description/z1_floating_base.xml`：带浮动基座的 MuJoCo 模型

**运行时请在仓库根目录执行**：

```bash
cd /path/to/drink_tea
python z1_whole_body_mpc_demo.py
```

---

## 2. 依赖一览

Python 版本建议：**Python 3.9–3.11**。

主要 Python 依赖：

- `numpy`
- `casadi`（MPC / 符号优化）
- `pinocchio` + `pinocchio.casadi`（刚体运动学 + CasADi 接口）
- `mujoco>=3.0.0`（物理仿真和可视化）

若你已经按 `pynebula/readme.md` 配好了开发环境，只需要再确认：

- 已安装 MuJoCo Python 包：`pip install mujoco`
- 已安装带 CasADi 支持的 Pinocchio（见下文第 3 节）

---

## 3. Pinocchio + CasADi 安装说明（重点）

本 Demo 需要：

- `import pinocchio as pin`
- `import pinocchio.casadi as cpin`

其中 `pinocchio.casadi` **只有在 Pinocchio 编译/打包时启用了 CasADi 支持** 时才会存在。下面给出**基于 Conda** 的安装方式。

### 3.1 使用 Conda 安装 `pinocchio` + `casadi`

推荐在 Conda 环境中一次性安装 `casadi` 和 `pinocchio`，环境名统一使用 `drink_tea`：

```bash
conda create -n drink_tea python=3.9
conda activate drink_tea

conda install -c conda-forge casadi pinocchio
pip install mujoco
```

注意事项：

- 请确保 `pinocchio` 构建时启用了 CasADi 支持（conda-forge 的二进制通常已开启）。
- `casadi` 与 `pinocchio` 必须版本兼容，否则可能导入失败。
- 建议参考本仓库的 `pynebula/requirements.txt` 中对 `casadi` 的版本建议（当前为 3.6.x）。

### 3.2 快速自检：确认 `pinocchio.casadi` 可用

安装完成后，可在终端中运行：

```bash
python - << 'PY'
import casadi
import pinocchio as pin
import pinocchio.casadi as cpin

print("casadi version:", casadi.__version__)
print("pinocchio version:", pin.__version__)
print("pinocchio.casadi OK:", cpin)
PY
```

若上述代码无异常输出，则说明 `pinocchio + casadi` 组合正确，`z1_whole_body_mpc_demo.py` 中的 FK/MPC 部分即可正常运行。

---

## 4. 推荐整体安装流程

下面给出一个从零开始、仅为跑本 Demo 准备的最小安装流程（基于 Conda）。

```bash
cd /path/to/drink_tea

# 1) 创建并激活 Conda 环境
conda create -n drink_tea python=3.9
conda activate drink_tea

# 2) 安装 Pinocchio + CasADi（Conda）
conda install -c conda-forge casadi pinocchio

# 3) 安装 MuJoCo 及基础依赖
pip install --upgrade pip
pip install "mujoco>=3.0.0" numpy
```

如果你还打算在同一环境中开发/运行 `pynebula`，也可以在同一个 Conda 环境中安装：

```bash
cd pynebula
pip install -r requirements.txt
pip install -e .
```

此时当前 Conda 环境中已经包含了 `casadi`、`pinocchio` 和 `mujoco`，可以直接回到仓库根目录运行 Demo。

---

## 5. 运行 Demo

环境准备好后，在仓库根目录执行：

```bash
cd /path/to/drink_tea
python z1_whole_body_mpc_demo.py
```

程序启动后会打开 MuJoCo Viewer，Z1 末端执行器沿随时间变化的椭圆轨迹运动，并在 `stdout` 中打印当前位置与参考位置之间的误差。

如果在远程/无显示环境下运行，可根据系统配置设置 `MUJOCO_GL`（例如 `egl` 或 `osmesa`），具体可参考 MuJoCo 官方文档。

---

## 6. 常见问题

- **报错：`ImportError: Pinocchio + pinocchio.casadi 未安装`**
  - 按第 3 节步骤使用 Conda 安装带 CasADi 支持的 `pinocchio`。
  - 用第 3.3 节的自检脚本确认 `pinocchio.casadi` 可导入。

- **报错：`ModuleNotFoundError: No module named 'mujoco'`**
  - 执行 `pip install mujoco`，并确保在同一个虚拟环境下运行脚本。

- **MuJoCo Viewer 无法启动 / 渲染报错**
  - 在 Linux 服务器等无显示环境下，尝试设置 `MUJOCO_GL=egl` 或 `osmesa`。
  - 确认 GPU/驱动及 OpenGL 环境配置正常。
