import doit


def task_start_backend():
    """Start the FastAPI backend server"""
    return {
        "actions": ["uvicorn main:app --reload --host 0.0.0.0 --port 5001"],
        "verbosity": 2,
        "params": [
            {
                "name": "backend_host",
                "short": "H",
                "default": "0.0.0.0",
                "type": str,
                "help": "Backend server host",
            },
            {
                "name": "backend_port",
                "short": "P",
                "default": 5001,
                "type": int,
                "help": "Backend server port",
            },
        ],
    }


def task_start_frontend():
    """Start the Streamlit frontend"""
    return {
        "actions": ["streamlit run app.py --server.port=8051"],
        "verbosity": 2,
        "params": [
            {
                "name": "frontend_port",
                "short": "P",
                "default": 5000,
                "type": int,
                "help": "Frontend server port",
            }
        ],
    }
