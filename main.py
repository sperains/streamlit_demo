import io
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 设置页面
st.set_page_config(
    page_title="海量数据动态可视化",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 标题
st.title("📊 海量数据动态可视化仪表板")
st.markdown("本应用演示如何在Streamlit中高效加载和可视化大规模数据集")

# 侧边栏配置
st.sidebar.header("数据生成配置")

# 数据生成参数
data_size = st.sidebar.selectbox(
    "选择数据量大小",
    [1e3, 1e4, 1e5, 1e6],
    format_func=lambda x: f"{int(x):,} 行",
    index=2,
)

num_categories = st.sidebar.slider("类别数量", 3, 20, 5)
noise_level = st.sidebar.slider("噪声水平", 0.0, 1.0, 0.1, 0.1)

# 图表配置
chart_type = st.sidebar.selectbox(
    "选择图表类型", ["折线图", "面积图", "散点图", "直方图", "热力图"]
)


# 缓存数据生成函数 - 避免重复生成相同数据
@st.cache_data(ttl=3600)
def generate_large_dataset(rows, categories, noise):
    """
    生成大规模模拟数据集 - 类型安全版本
    """
    st.sidebar.info(f"生成 {int(rows):,} 行数据...")

    # 创建时间序列
    dates = pd.date_range("2020-01-01", periods=int(rows), freq="H")

    # 生成基础数据框架
    df = pd.DataFrame(
        {"timestamp": dates, "base_trend": np.linspace(0, 100, len(dates))}
    )

    # 为每个类别生成数据
    for i in range(categories):
        # 添加季节性成分
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 7))
        # 添加随机噪声
        random_noise = noise * 50 * np.random.randn(len(dates))
        # 添加类别特定的趋势
        category_trend = np.linspace(0, (i + 1) * 20, len(dates))

        # 确保生成数值类型数据
        df[f"category_{i + 1}"] = (
            df["base_trend"] + seasonal + random_noise + category_trend
        ).astype(float)  # 明确转换为float类型

    # 添加异常值 - 类型安全的方式
    outlier_indices = np.random.choice(len(df), size=int(0.01 * len(df)), replace=False)
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for idx in outlier_indices:
        if len(numeric_columns) > 1:  # 确保有数值列可供选择
            # 从数值列中随机选择一个（排除timestamp相关的列）
            available_cols = [
                col for col in numeric_columns if col not in ["timestamp", "base_trend"]
            ]
            if available_cols:
                col_name = np.random.choice(available_cols)
                current_value = df.loc[idx, col_name]
                # 安全地进行乘法运算
                df.loc[idx, col_name] = current_value * 2

    # 添加分类变量
    df["group"] = np.random.choice(["A", "B", "C"], len(df))

    return df


# 数据聚合函数 - 用于处理大规模数据可视化
@st.cache_data
def aggregate_data(df, aggregation_level="H"):
    """
    根据时间粒度聚合数据，提高大规模数据可视化性能
    """
    df_agg = (
        df.set_index("timestamp")
        .resample(aggregation_level)
        .mean(numeric_only=True)
        .reset_index()
    )
    return df_agg


# 生成数据
with st.spinner("正在生成数据..."):
    df = generate_large_dataset(data_size, num_categories, noise_level)

# 显示数据基本信息
col1, col2, col3, col4 = st.columns(4)
col1.metric("数据行数", f"{len(df):,}")
col2.metric("数据列数", len(df.columns))
col3.metric(
    "时间范围", f"{df['timestamp'].min().date()} 至 {df['timestamp'].max().date()}"
)
col4.metric("内存使用", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

# 数据预览
st.subheader("数据预览")
with st.expander("查看原始数据"):
    st.dataframe(df.head(1000), use_container_width=True)

# 动态图表区域
st.subheader("动态数据可视化")

# 选择要显示的类别
selected_categories = st.multiselect(
    "选择要显示的类别",
    options=[f"category_{i + 1}" for i in range(num_categories)],
    default=[f"category_{i + 1}" for i in range(min(3, num_categories))],
)

# 时间范围选择器
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "开始日期",
        value=df["timestamp"].min().date(),
        min_value=df["timestamp"].min().date(),
        max_value=df["timestamp"].max().date(),
    )
with col2:
    end_date = st.date_input(
        "结束日期",
        value=df["timestamp"].max().date(),
        min_value=df["timestamp"].min().date(),
        max_value=df["timestamp"].max().date(),
    )

# 过滤数据
start_dt = pd.Timestamp(start_date)
end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
filtered_df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)]

# 根据数据量自动选择聚合级别
if len(filtered_df) > 10000:
    agg_level = "6H"  # 每6小时聚合
elif len(filtered_df) > 5000:
    agg_level = "2H"  # 每2小时聚合
else:
    agg_level = "H"  # 每小时聚合

# 显示聚合信息
st.info(
    f"数据点过多 ({len(filtered_df):,} 行)，已自动聚合到 {agg_level} 级别以提高性能"
)

# 聚合数据
agg_df = aggregate_data(filtered_df, agg_level)

# 创建图表
tab1, tab2, tab3 = st.tabs(["主要图表", "统计分析", "数据导出"])

with tab1:
    if selected_categories:
        if chart_type == "折线图":
            fig = px.line(
                agg_df,
                x="timestamp",
                y=selected_categories,
                title=f"时间序列数据 ({agg_level} 聚合)",
                labels={"value": "数值", "timestamp": "时间"},
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "面积图":
            fig = px.area(
                agg_df,
                x="timestamp",
                y=selected_categories,
                title=f"面积图 ({agg_level} 聚合)",
                labels={"value": "数值", "timestamp": "时间"},
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "散点图":
            if len(selected_categories) >= 2:
                fig = px.scatter(
                    agg_df,
                    x=selected_categories[0],
                    y=selected_categories[1],
                    color="group" if "group" in agg_df.columns else None,
                    title="散点图",
                    size_max=10,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("散点图需要至少选择两个类别")

        elif chart_type == "直方图":
            fig = px.histogram(
                agg_df,
                x=selected_categories[0],
                nbins=50,
                title=f"{selected_categories[0]} 分布直方图",
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "热力图":
            # 创建相关性热力图
            corr_matrix = agg_df[selected_categories].corr()
            fig = px.imshow(
                corr_matrix,
                title="类别相关性热力图",
                color_continuous_scale="RdBu_r",
                aspect="auto",
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("请至少选择一个类别进行可视化")

with tab2:
    st.subheader("数据统计分析")

    if selected_categories:
        # 描述性统计
        st.write("描述性统计:")
        desc_stats = filtered_df[selected_categories].describe()
        st.dataframe(desc_stats, use_container_width=True)

        # 相关性矩阵
        st.write("相关性矩阵:")
        correlation = filtered_df[selected_categories].corr()
        st.dataframe(
            correlation.style.background_gradient(cmap="coolwarm"),
            use_container_width=True,
        )

        # 添加一些额外的统计图表
        col1, col2 = st.columns(2)

        with col1:
            # 箱线图
            fig_box = px.box(
                filtered_df.melt(
                    value_vars=selected_categories,
                    var_name="category",
                    value_name="value",
                ),
                x="category",
                y="value",
                title="类别分布箱线图",
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with col2:
            # 滚动平均值
            if len(selected_categories) > 0:
                rolling_avg = (
                    filtered_df[selected_categories[0]].rolling(window=24).mean()
                )
                fig_roll = go.Figure()
                fig_roll.add_trace(
                    go.Scatter(
                        x=filtered_df["timestamp"],
                        y=filtered_df[selected_categories[0]],
                        name="原始数据",
                        opacity=0.3,
                    )
                )
                fig_roll.add_trace(
                    go.Scatter(
                        x=filtered_df["timestamp"], y=rolling_avg, name="24小时滚动平均"
                    )
                )
                fig_roll.update_layout(title=f"{selected_categories[0]} 滚动平均")
                st.plotly_chart(fig_roll, use_container_width=True)

with tab3:
    st.subheader("数据导出")

    # 导出选项
    export_format = st.radio("选择导出格式", ["CSV", "Parquet", "JSON"])

    # 根据选择创建数据
    if export_format == "CSV":
        csv = df.to_csv(index=False)
        st.download_button(
            label="下载 CSV 文件",
            data=csv,
            file_name=f"large_dataset_{int(data_size)}_rows.csv",
            mime="text/csv",
        )
    elif export_format == "Parquet":
        # 使用BytesIO创建内存中的Parquet文件
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        st.download_button(
            label="下载 Parquet 文件",
            data=buffer.getvalue(),
            file_name=f"large_dataset_{int(data_size)}_rows.parquet",
            mime="application/octet-stream",
        )
    elif export_format == "JSON":
        json_str = df.head(10000).to_json(
            orient="records", date_format="iso"
        )  # 限制JSON大小
        st.download_button(
            label="下载 JSON 文件 (前10,000行)",
            data=json_str,
            file_name="large_dataset_sample.json",
            mime="application/json",
        )

    # 显示导出统计信息
    st.info(f"完整数据集包含 {len(df):,} 行，{len(df.columns)} 列")

# 性能监控部分
st.sidebar.markdown("---")
st.sidebar.subheader("性能监控")

# 显示数据加载时间
if st.sidebar.button("刷新性能指标"):
    start_time = time.time()
    _ = generate_large_dataset(data_size, num_categories, noise_level)
    load_time = time.time() - start_time

    st.sidebar.metric("数据生成时间", f"{load_time:.2f} 秒")
    st.sidebar.metric("数据点数量", f"{len(df) * num_categories:,}")

# 使用说明
with st.expander("使用说明"):
    st.markdown("""
    ### 使用指南
    
    1. **数据生成配置**:
       - 调整侧边栏参数控制生成的数据规模和特性
       - 数据量越大，生成时间越长，但Streamlit的缓存机制会优化重复生成
    
    2. **可视化优化**:
       - 应用会自动检测数据量并选择合适的聚合级别
       - 使用Plotly图表库提供交互式体验
    
    3. **性能提示**:
       - 对于超大数据集(>1M行)，考虑使用更粗的时间聚合
       - 只选择需要的类别进行可视化以提高性能
       - 利用缓存避免重复计算
    """)

# 底部信息
st.markdown("---")
st.markdown(
    "📈 使用Streamlit高效处理海量数据可视化 | 优化策略: 数据聚合、缓存、选择性加载"
)
