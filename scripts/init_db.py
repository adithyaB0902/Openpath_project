from openpath.config import get_settings
from openpath.db import init_db

if __name__ == "__main__":
    s = get_settings()
    init_db(s.db_path)
    print(f"Database ready at {s.db_path}")
