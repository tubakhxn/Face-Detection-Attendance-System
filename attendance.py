import pandas as pd
import os
from datetime import datetime

def mark_attendance(name, emp_id):
    date = datetime.now().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H:%M:%S')
    csv_file = 'attendance.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=['Name', 'ID', 'Date', 'Time'])
    # Check if already marked today
    if ((df['Name'] == name) & (df['ID'] == emp_id) & (df['Date'] == date)).any():
        return False
    df = pd.concat([df, pd.DataFrame([[name, emp_id, date, time]], columns=['Name', 'ID', 'Date', 'Time'])], ignore_index=True)
    df.to_csv(csv_file, index=False)
    return True

def view_attendance_log():
    csv_file = 'attendance.csv'
    if not os.path.exists(csv_file):
        print("No attendance log found.")
        return
    df = pd.read_csv(csv_file)
    print("\n--- Attendance Log ---")
    print(df.tail(30).to_string(index=False))
