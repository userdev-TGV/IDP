#!/bin/bash

echo "Iniciando Streamlit con streamlit_run.py..."
streamlit run app.py --server.port=8000 --server.address=0.0.0.0
