from locust import HttpUser, task, constant

class ApiUser(HttpUser):
    host = "http://localhost"
    wait_time = constant(1)

    @task
    def call_api(self):
        self.client.get("/")