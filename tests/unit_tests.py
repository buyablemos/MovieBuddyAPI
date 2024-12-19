import pytest
from flask import Flask
from flask.testing import FlaskClient
from PythonFiles.app import app


@pytest.fixture
def client() -> FlaskClient:
    """Funkcja testowa, która konfiguruje i zwraca klienta do testowania."""
    with app.test_client() as client:
        yield client


def test_compute_sha256():
    from PythonFiles.app import compute_sha256
    input_string = "dawid"
    expected_hash = "34b37f1e6f801286d05a7818adf5c4f7542fff8a5d719bb3d832b7fa3bae4363"
    assert compute_sha256(input_string) == expected_hash
    # Testuje funkcję haszującą SHA-256


def test_register_user():
    from PythonFiles.db import Database
    from PythonFiles.app import compute_sha256
    db = Database()

    db.conn.start_transaction()

    try:
        db.register_user(
            username="testuser",
            email="testuser@example.com",
            password=compute_sha256("hashedpassword"),
            gender="M"
        )

        user = db.get_registered_user_by_username("testuser")
        db.delete_user_by_username("testuser")
        db.conn.rollback()
        assert user[1] == "testuser"
        assert user[2] == "testuser@example.com"
        assert user[3] == compute_sha256("hashedpassword")
        assert user[4] == "M"

    except Exception as e:
        db.conn.rollback()
        raise e

    finally:
        db.conn.rollback()
    # Testuje proces rejestracji użytkownika


def test_register_user_with_duplicate_email():
    from PythonFiles.db import Database
    db = Database()

    db.register_user(
        username="user1",
        email="duplicate@example.com",
        password="password1",
        gender="M"
    )
    try:
        db.register_user(
            username="user2",
            email="duplicate@example.com",
            password="password2",
            gender="F"
        )
        assert False, "Expected ValueError for duplicate email"
    except ValueError as e:
        assert str(e) == "Username or email already exists"
    finally:
        db.delete_user_by_email("duplicate@example.com")
    # Testuje rejestrację z duplikatem e-mail


def test_register_user_with_duplicate_username():
    from PythonFiles.db import Database
    db = Database()

    db.register_user(
        username="user1",
        email="oldemail@example.com",
        password="password1",
        gender="M"
    )
    try:
        db.register_user(
            username="user1",
            email="newemail@example.com",
            password="password2",
            gender="F"
        )
        assert False, "Expected ValueError for duplicate email"
    except ValueError as e:
        assert str(e) == "Username or email already exists"
    finally:
        db.delete_user_by_username("user1")
    # Testuje rejestrację z duplikatem nazwy użytkownika


def test_register_endpoint_user(client):
    response1 = client.post("/register", json={
        "username": "user1",
        "email": "duplicate@example.com",
        "password": "password1",
        "gender": "M"
    })

    assert response1.status_code == 201
    assert response1.json["message"] == 'User registered successfully!'
# Testuje rejestrację użytkownika przez endpoint

def test_login_endpoint_user(client):
    from PythonFiles.db import Database

    db = Database()

    response2 = client.post("/login", json={
        "username": "user1",
        "password": "password1"
    })

    db.delete_user_by_username("user1")
    assert response2.status_code == 201
    assert response2.json["message"] == "User login successfully!"
    # Testuje logowanie użytkownika przez endpoint


def test_login_user_invalid(client):
    from PythonFiles.db import Database
    from PythonFiles.app import compute_sha256
    db = Database()

    db.register_user(
        username="user1",
        email="duplicate@example.com",
        password=compute_sha256("password1"),
        gender="M"
    )
    response = client.post("/login", json={
        "username": "user1",
        "password": "passwordexample"
    })

    db.delete_user_by_username("user1")
    assert response.status_code == 400
    assert response.json["error"] == "Invalid username or password."
    # Testuje logowanie z niepoprawnym hasłem


def test_recommend_on_history_kNN_CF(client):
    params = {
        'user_id': 1,
        'n_recommend': 7
    }

    response = client.get('/recommend_on_history_kNN_CF', query_string=params)

    assert response.status_code == 200

    assert "data" in response.json

    assert len(response.json["data"]) == 7
    # Testuje rekomendacje na podstawie historii przy użyciu kNN CF


def test_recommend_on_history_kNN_CBF(client):
    params = {
        'user_id': 1,
        'n_recommend': 7
    }

    response = client.get('/recommend_on_history_kNN_CBF', query_string=params)

    assert response.status_code == 200

    assert "data" in response.json

    assert len(response.json["data"]) == 7
    # Testuje rekomendacje na podstawie historii przy użyciu kNN CBF


def test_recommend_on_movie_kNN_CBF(client):
    params = {
        'movieId': 1,
        'n_recommend': 7
    }

    response = client.get('/recommend_on_movie_kNN_CBF', query_string=params)

    assert response.status_code == 200

    assert "data" in response.json

    assert len(response.json["data"]) == 7
    # Testuje rekomendacje na podstawie filmu przy użyciu kNN CBF


def test_recommend_on_movie_kNN_CF(client):
    params = {
        'movieId': 1,
        'n_recommend': 7
    }

    response = client.get('/recommend_on_movie_kNN_CF', query_string=params)

    assert response.status_code == 200

    assert "data" in response.json

    assert len(response.json["data"]) == 7
    # Testuje rekomendacje na podstawie filmu przy użyciu kNN CF


def test_add_user_rating(client):
    data = {
        'userId': 657,
        'movieId': 100,
        'rating': 4.5
    }

    response = client.post('/add-rating', json=data)

    assert response.status_code == 200
    assert response.json['success'] is True
    assert response.json['message'] == 'Rating added successfully'
    # Testuje dodanie oceny użytkownika


def test_delete_user_rating(client):
    user_id = 657
    movie_id = 100

    response = client.delete(f'/{user_id}/ratings/{movie_id}')

    assert response.status_code == 200
    assert response.json['success'] is True
    assert response.json['message'] == 'Rating deleted successfully'
    # Testuje usunięcie oceny użytkownika


def test_delete_non_existent_rating(client):
    user_id = 657
    movie_id = 999999999999999

    response = client.delete(f'/{user_id}/ratings/{movie_id}')

    assert response.status_code == 404
    assert response.json['success'] is False
    assert response.json['message'] == 'Rating not found'
    # Testuje usunięcie nieistniejącej oceny przez użytkownika


def test_add_rating_missing_parameters(client):
    data = {
        'userId': 657,
    }
    response = client.post('/add-rating', json=data)

    assert response.status_code == 400
    assert response.json['success'] is False
    assert response.json["error"] == 'Missing parameters'
    # Testuje, czy aplikacja poprawnie reaguje na zapytania, które nie zawierają wymaganych parametrów



