import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Heatmaps of AFR and Pulse Width vs MAP and RPM with Sample Counts")

uploaded_file = st.file_uploader("Upload OBD2 log CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdown("### Upload Field Mapping CSV File")
    mapping_file = st.file_uploader("Upload CSV with field mappings (original,new)", type=["csv"])

    if mapping_file is not None:
        mapping_df = pd.read_csv(mapping_file)
        column_mapping = dict(zip(mapping_df['original'], mapping_df['new']))
        df.rename(columns=column_mapping, inplace=True)

        required_fields = ['RPM', 'MAP', 'AFR_specified', 'IPW_ms']
        missing_fields = [col for col in required_fields if col not in df.columns]
        if missing_fields:
            st.error(f"Missing required fields after mapping: {', '.join(missing_fields)}")
        else:
            df = df.dropna(subset=required_fields)
            df['RPM'] = pd.to_numeric(df['RPM'], errors='coerce')
            df['MAP'] = pd.to_numeric(df['MAP'], errors='coerce')
            df['AFR_specified'] = pd.to_numeric(df['AFR_specified'], errors='coerce')
            df['IPW_ms'] = pd.to_numeric(df['IPW_ms'], errors='coerce')
            df = df[(df['RPM'] > 0) & (df['MAP'] > 0) & (df['AFR_specified'] > 0)]

            # Calculate actual AFR
            AFR_stoich = 14.7
            df['AFR_actual'] = df['AFR_specified'] * AFR_stoich

            # Create bins for MAP and RPM
            map_bins = np.arange(0, df['MAP'].max() + 50, 50)
            rpm_bins = np.arange(0, df['RPM'].max() + 500, 500)
            df['MAP_bin'] = pd.cut(df['MAP'], bins=map_bins)
            df['RPM_bin'] = pd.cut(df['RPM'], bins=rpm_bins)

            # Bin centers for labels
            df['MAP_bin_center'] = df['MAP_bin'].apply(lambda x: (x.left + x.right) / 2 if pd.notna(x) else np.nan)
            df['RPM_bin_center'] = df['RPM_bin'].apply(lambda x: (x.left + x.right) / 2 if pd.notna(x) else np.nan)

            # Drop rows with NA in bin centers
            heatmap_df = df.dropna(subset=['MAP_bin_center', 'RPM_bin_center', 'AFR_actual', 'IPW_ms'])

            def create_heatmap_with_text(z_values, x_labels, y_labels, title, colorscale, text_colors):
                # z_values: 2D numpy array or DataFrame values (masked with NaNs where no data)
                # text_colors: dict with keys 'high' and 'low' for text color based on background
                
                z_mask = ~np.isnan(z_values)
                text = np.where(z_mask, np.round(z_values, 2).astype(str), "")

                fig = go.Figure(
                    data=go.Heatmap(
                        z=z_values,
                        x=x_labels,
                        y=y_labels,
                        colorscale=colorscale,
                        colorbar=dict(title=title),
                        text=text,
                        texttemplate="%{text}",
                        textfont={"color": "white"},
                        hoverongaps=False,
                    )
                )

                # Adjust text color based on z values for readability:
                # Plotly does not support dynamic text color in heatmaps, so all white here.
                # Could add annotations manually for full control if needed.

                fig.update_layout(
                    title=title,
                    xaxis_title="RPM (bin center)",
                    yaxis_title="MAP (kPa bin center)",
                    height=600,
                    width=900
                )
                return fig

            st.markdown("### AFR (Actual) Heatmap and Sample Count")

            # AFR mean and count pivot tables
            pivot_afr = heatmap_df.pivot_table(
                values='AFR_actual',
                index='MAP_bin_center',
                columns='RPM_bin_center',
                aggfunc='mean'
            )
            pivot_afr_count = heatmap_df.pivot_table(
                values='AFR_actual',
                index='MAP_bin_center',
                columns='RPM_bin_center',
                aggfunc='count'
            )

            # Align shapes
            pivot_afr_count = pivot_afr_count.reindex(index=pivot_afr.index, columns=pivot_afr.columns)
            pivot_afr_masked = pivot_afr.where(pivot_afr_count > 0)

            col1, col2 = st.columns(2)

            with col1:
                fig_afr = create_heatmap_with_text(
                    pivot_afr_masked.values,
                    pivot_afr.columns,
                    pivot_afr.index,
                    "AFR Actual",
                    "Viridis",
                    text_colors={"high": "white", "low": "black"}
                )
                st.plotly_chart(fig_afr, use_container_width=True)

            with col2:
                fig_afr_count = create_heatmap_with_text(
                    pivot_afr_count.values,
                    pivot_afr_count.columns,
                    pivot_afr_count.index,
                    "Sample Count (AFR)",
                    "Blues",
                    text_colors={"high": "black", "low": "black"}
                )
                st.plotly_chart(fig_afr_count, use_container_width=True)

            st.markdown("### Logged Pulse Width (IPW_ms) Heatmap and Sample Count")

            # IPW mean and count pivot tables
            pivot_ipw = heatmap_df.pivot_table(
                values='IPW_ms',
                index='MAP_bin_center',
                columns='RPM_bin_center',
                aggfunc='mean'
            )
            pivot_ipw_count = heatmap_df.pivot_table(
                values='IPW_ms',
                index='MAP_bin_center',
                columns='RPM_bin_center',
                aggfunc='count'
            )

            # Align shapes
            pivot_ipw_count = pivot_ipw_count.reindex(index=pivot_ipw.index, columns=pivot_ipw.columns)
            pivot_ipw_masked = pivot_ipw.where(pivot_ipw_count > 0)

            col3, col4 = st.columns(2)

            with col3:
                fig_ipw = create_heatmap_with_text(
                    pivot_ipw_masked.values,
                    pivot_ipw.columns,
                    pivot_ipw.index,
                    "Pulse Width (ms)",
                    "Viridis",
                    text_colors={"high": "white", "low": "black"}
                )
                st.plotly_chart(fig_ipw, use_container_width=True)

            with col4:
                fig_ipw_count = create_heatmap_with_text(
                    pivot_ipw_count.values,
                    pivot_ipw_count.columns,
                    pivot_ipw_count.index,
                    "Sample Count (Pulse Width)",
                    "Blues",
                    text_colors={"high": "black", "low": "black"}
                )
                st.plotly_chart(fig_ipw_count, use_container_width=True)

    else:
        st.warning("Please upload a field mapping CSV file to proceed.")
else:
    st.info("Please upload a CSV file to proceed.")
