import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder as SklearnOneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
import shap
import os
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

st.set_page_config(page_title='cGAN-Powered Intrusion Detection System', page_icon='🛡️')
st.title('🛡️cGAN-Powered Intrusion Detection System')
st.markdown("---")

# 1. Simulate IoT Intrusion Data
st.header('1. Simulate IoT Intrusion Data')
n_samples = st.slider('Number of synthetic samples', 1, 10000, 1000)
if st.button('Generate Synthetic Data'):
    with st.spinner('Generating...'):
        cGAN_RT_IDS_synthesizer = CTGANSynthesizer.load("cGAN_IDS_synthesizer.pkl")
        synthetic_data = cGAN_RT_IDS_synthesizer.sample(n_samples)
        synthetic_data.to_csv('synthetic_data.csv', index=False)
        st.success('Generated!')
        with open('synthetic_data.csv', 'rb') as f:
            st.download_button('Download Simulated Data CSV', f, file_name='synthetic_data.csv')

# 2. Data Pre-processing
st.header('2. Data Pre-processing')
uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'], key='file_uploader')
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if st.button('Run Diagnostic'):
        try:
            with st.spinner('Running diagnostic...'):
                if not os.path.exists('original_df.csv'):
                    st.error('original_df.csv not found! Please ensure the file is present in the working directory.')
                else:
                    orginal_df = pd.read_csv('original_df.csv', index_col=0)
                    if set(orginal_df.columns) != set(data.columns):
                        st.error('Column mismatch between original and uploaded data! Please check your file.')
                    else:
                        org_sample = orginal_df.sample(n=min(1000, len(orginal_df)), random_state=42)
                        data_sample = data.sample(n=min(1000, len(data)), random_state=42)
                        metadata = SingleTableMetadata()
                        metadata.detect_from_dataframe(orginal_df)
                        diagnostic = run_diagnostic(org_sample, data_sample, metadata, verbose=False)
                        fig = diagnostic.get_visualization(property_name='Data Validity')
                        st.write(f'Diagnostic Fig (type: {type(fig)})')
                        st.plotly_chart(fig)
                        st.write('Diagnostic Report:', diagnostic.get_details(property_name='Data Validity'))
        except Exception as e:
            st.error(f"Diagnostic failed: {e}")

    attacks = ['DOS_SYN_Hping', 'Thing_Speak', 'ARP_poisioning', 'MQTT_Publish', 'NMAP_UDP_SCAN', 'NMAP_XMAS_TREE_SCAN', 'Wipro_bulb']
    data = data[data['Attack_type'].isin(attacks)]

    # Data preprocessing
    target_col = "Attack_type"
    from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
    supported_cols = ["active_avg", "service_radius", "id_resp_p", "fwd_pkts_per_sec", "fwd_pkts_payload_avg", "id_orig_p", "flow_pkts_payload_tot", "service_dns", "fwd_header_size_tot", "service_dhcp", "flow_pkts_payload_min", "bwd_pkts_per_sec", "flow_pkts_per_sec", "active_min", "service_http", "fwd_header_size_max", "flow_pkts_payload_max", "bwd_pkts_payload_avg", "proto_tcp", "flow_SYN_flag_count", "service_mqtt", "fwd_header_size_min", "bwd_header_size_max", "fwd_pkts_payload_min", "fwd_iat_min", "service_unspec", "proto_udp", "flow_pkts_payload_avg", "fwd_URG_flag_count", "proto_icmp", "service_ssl", "flow_iat_min", "fwd_pkts_payload_tot", "fwd_last_window_size", "service_irc", "flow_FIN_flag_count", "payload_bytes_per_second", "fwd_init_window_size", "fwd_pkts_payload_max", "service_ntp"]
    col_selector = ColumnSelector(supported_cols)

    # Boolean columns
    bool_imputers = []
    bool_pipeline = Pipeline(steps=[
        ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
        ("imputers", ColumnTransformer(bool_imputers, remainder="passthrough")),
        ("onehot", SklearnOneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
    bool_transformers = [("boolean", bool_pipeline, ["proto_udp", "service_http", "service_irc", "fwd_URG_flag_count", "service_radius", "proto_icmp", "service_ssl", "service_dns", "proto_tcp", "service_mqtt", "service_dhcp", "service_ntp", "service_unspec"])]

    # Numerical columns
    from sklearn.impute import SimpleImputer
    num_imputers = []
    num_imputers.append(("impute_mean", SimpleImputer(), ["active_avg", "active_min", "bwd_header_size_max", "bwd_pkts_payload_avg", "bwd_pkts_per_sec", "flow_FIN_flag_count", "flow_SYN_flag_count", "flow_iat_min", "flow_pkts_payload_avg", "flow_pkts_payload_max", "flow_pkts_payload_min", "flow_pkts_payload_tot", "flow_pkts_per_sec", "fwd_URG_flag_count", "fwd_header_size_max", "fwd_header_size_min", "fwd_header_size_tot", "fwd_iat_min", "fwd_init_window_size", "fwd_last_window_size", "fwd_pkts_payload_avg", "fwd_pkts_payload_max", "fwd_pkts_payload_min", "fwd_pkts_payload_tot", "fwd_pkts_per_sec", "id_orig_p", "id_resp_p", "payload_bytes_per_second", "proto_icmp", "proto_tcp", "proto_udp", "service_dhcp", "service_dns", "service_http", "service_irc", "service_mqtt", "service_ntp", "service_radius", "service_ssl", "service_unspec"]))
    numerical_pipeline = Pipeline(steps=[
        ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
        ("imputers", ColumnTransformer(num_imputers)),
        ("standardizer", StandardScaler()),
    ])
    numerical_transformers = [("numerical", numerical_pipeline, ["active_avg", "id_resp_p", "fwd_pkts_payload_avg", "fwd_pkts_per_sec", "id_orig_p", "service_radius", "flow_pkts_payload_tot", "service_dns", "fwd_header_size_tot", "service_dhcp", "flow_pkts_payload_min", "bwd_pkts_per_sec", "flow_pkts_per_sec", "active_min", "service_http", "fwd_header_size_max", "flow_pkts_payload_max", "bwd_pkts_payload_avg", "flow_SYN_flag_count", "proto_tcp", "fwd_header_size_min", "service_mqtt", "bwd_header_size_max", "fwd_pkts_payload_min", "fwd_iat_min", "service_unspec", "flow_pkts_payload_avg", "proto_udp", "fwd_URG_flag_count", "proto_icmp", "service_ssl", "flow_iat_min", "fwd_pkts_payload_tot", "fwd_last_window_size", "service_irc", "flow_FIN_flag_count", "payload_bytes_per_second", "fwd_init_window_size", "fwd_pkts_payload_max", "service_ntp"])]

    transformers = bool_transformers + numerical_transformers
    preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0) 

    # Separate target column from features
    y = data[target_col]
    data = data.drop([target_col], axis=1)

    pipeline_val = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
    ])
    pipeline_val.fit(data, y)

    cast_type_feature_names = ["proto_udp", "service_http", "service_irc", "fwd_URG_flag_count", "service_radius", "proto_icmp", "service_ssl", "service_dns", "proto_tcp", "service_mqtt", "service_dhcp", "service_ntp", "service_unspec"]
    onehot_feature_names = pipeline_val.named_steps['preprocessor'].transformers_[0][1].named_steps['onehot'].get_feature_names_out(cast_type_feature_names)
    numerical_feature_names = ["active_avg", "id_resp_p", "fwd_pkts_payload_avg", "fwd_pkts_per_sec", "id_orig_p", "service_radius", "flow_pkts_payload_tot", "service_dns", "fwd_header_size_tot", "service_dhcp", "flow_pkts_payload_min", "bwd_pkts_per_sec", "flow_pkts_per_sec", "active_min", "service_http", "fwd_header_size_max", "flow_pkts_payload_max", "bwd_pkts_payload_avg", "flow_SYN_flag_count", "proto_tcp", "fwd_header_size_min", "service_mqtt", "bwd_header_size_max", "fwd_pkts_payload_min", "fwd_iat_min", "service_unspec", "flow_pkts_payload_avg", "proto_udp", "fwd_URG_flag_count", "proto_icmp", "service_ssl", "flow_iat_min", "fwd_pkts_payload_tot", "fwd_last_window_size", "service_irc", "flow_FIN_flag_count", "payload_bytes_per_second", "fwd_init_window_size", "fwd_pkts_payload_max", "service_ntp"]
    all_feature_names = list(onehot_feature_names) + numerical_feature_names

    data_processed = pipeline_val.transform(data)
    model = joblib.load('lightGBM_multiclass_classifier.joblib')
    data_processed = pd.DataFrame(data_processed, columns=model.feature_name_)

    st.write(f"Shape of data_processed: {data_processed.shape}")
    st.write(f"Model expects: {len(model.feature_name_)} features")

    # 3. Model Inference
    st.header('3. Model Inference')
    if st.button('Run Model Inference'):
        with st.spinner('Loading model and predicting...'):
            model = joblib.load('lightGBM_multiclass_classifier.joblib')
            predictions = model.predict(data_processed)
            sample_prediction = pd.DataFrame({
                'Actual Labels': y,
                'Predictions': predictions
            })
            st.write('Sample Predictions:', sample_prediction.head())

            # Calculate and display additional metrics
            st.subheader('Evaluation Metrics')
            accuracy = accuracy_score(y, predictions)
            precision = precision_score(y, predictions, average='weighted')
            recall = recall_score(y, predictions, average='weighted')
            f1 = f1_score(y, predictions, average='weighted')

            st.write(f"Accuracy: {accuracy:.4f}")
            st.write(f"Precision (weighted): {precision:.4f}")
            st.write(f"Recall (weighted): {recall:.4f}")
            st.write(f"F1-Score (weighted): {f1:.4f}")

            st.subheader('Classification Report')
            label_dict = {
                1: "DOS_SYN_Hping",
                5: "Thing_Speak",
                0: "ARP_poisioning",
                2: "MQTT_Publish",
                4: "NMAP_XMAS_TREE_SCAN",
                3: "NMAP_UDP_SCAN",
                6: "Wipro_bulb"
            }
            labels = [label_dict[i] for i in sorted(label_dict.keys())]
            report = classification_report(y, predictions, target_names=labels, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            st.subheader('Normalized Confusion Matrix')
            cm = confusion_matrix(y, predictions)
            cm_normalized = normalize(cm, axis=1, norm='l1')
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f',
                             xticklabels=labels, yticklabels=labels)
            label_font = {'size': '12'}
            ax.set_xlabel('Predicted labels', fontdict=label_font)
            ax.set_ylabel('True labels', fontdict=label_font)
            ax.set_title('Normalized Confusion Matrix', fontdict={'size': '15'})
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels, rotation=0)
            st.pyplot(plt)

        
st.warning('Simulate IoT intrusion data to get started!')
