import json
from PythonFiles.db import Database


class LastUserInfo:
    @staticmethod
    def save_last_trained_user(model_name):
        db = Database()
        user_id = db.get_last_user_id()
        filename = '../last_trained_user.json'

        try:
            try:
                with open(filename, 'r') as file:
                    existing_data = json.load(file)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = {}  # Jeśli plik nie istnieje lub jest pusty

            existing_data[f"last_trained_user_id_{model_name}"] = user_id

            # Zapisanie zaktualizowanych danych do pliku
            with open(filename, 'w') as file:
                json.dump(existing_data, file, indent=2)
                print("Saved last trained user: "+str(user_id)+", model: "+model_name)
            return user_id
        except (IOError, OSError) as e:
            print(f"Problem with saving last trained id: {e}")

        return None

    @staticmethod
    def read_last_trained_user(model_name):
        filename = '../last_trained_user.json'
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                data = data.get(f"last_trained_user_id_{model_name}")
            return data
        except (IOError, OSError) as e:
            print(f"Problem with reading last trained id: {e}")
            return None
