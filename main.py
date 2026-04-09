import os
import sys
import subprocess
from utils import install_requirements
from face_capture import add_employee
from train_model import train_model
from recognizer import start_attendance_system
from attendance import view_attendance_log

def menu():
    while True:
        print("\n==== Face Recognition Attendance System ====")
        print("1. Add New Employee")
        print("2. Train Model (generate encodings)")
        print("3. Start Attendance System")
        print("4. View Attendance Log")
        print("5. Exit")
        choice = input("Select an option: ")
        if choice == '1':
            add_employee()
        elif choice == '2':
            train_model()
        elif choice == '3':
            start_attendance_system()
        elif choice == '4':
            view_attendance_log()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    install_requirements()
    menu()
