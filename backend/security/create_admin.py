"""
AwaazDesk - Secure Administrative User Creation CLI Utility
Allows administrators to securely provision or update admin/officer accounts
with Bcrypt password hashing.

Usage:
    python -m backend.security.create_admin
    python -m backend.security.create_admin --username officer_sharma --role officer
"""
import sys
import getpass
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import init_db, get_user_by_username, create_user, get_db_connection
from backend.security.auth import hash_password

def main():
    parser = argparse.ArgumentParser(description="Create or update an administrative user for AwaazDesk.")
    parser.add_argument("--username", "-u", type=str, help="Username for the administrative account")
    parser.add_argument("--password", "-p", type=str, help="Password (if omitted, will be prompted securely)")
    parser.add_argument("--role", "-r", type=str, choices=["admin", "officer"], default="admin", help="Role (default: admin)")
    parser.add_argument("--email", "-e", type=str, default=None, help="Contact email address")
    parser.add_argument("--full-name", "-n", type=str, default=None, help="Officer full name")

    args = parser.parse_args()

    # Ensure database schema is present
    init_db(seed_sample_data=False)

    username = args.username
    if not username:
        try:
            username = input("Enter administrative username: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(1)

    if not username:
        print("Error: Username cannot be empty.")
        sys.exit(1)

    password = args.password
    if not password:
        try:
            password = getpass.getpass(f"Enter password for '{username}': ").strip()
            confirm_pwd = getpass.getpass("Confirm password: ").strip()
            if password != confirm_pwd:
                print("Error: Passwords do not match.")
                sys.exit(1)
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(1)

    if len(password) < 6:
        print("Error: Password must be at least 6 characters long.")
        sys.exit(1)

    hashed = hash_password(password)

    existing = get_user_by_username(username)
    if existing:
        print(f"User '{username}' already exists. Updating role and password hash...")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE users
        SET hashed_password = ?, role = ?, is_active = 1,
            email = COALESCE(?, email), full_name = COALESCE(?, full_name)
        WHERE username = ?
        """, (hashed, args.role, args.email, args.full_name, username))
        conn.commit()
        conn.close()
        print(f"SUCCESS: User '{username}' updated with role '{args.role}'. Password securely hashed with Bcrypt.")
    else:
        created = create_user(
            username=username,
            email=args.email,
            full_name=args.full_name,
            hashed_password=hashed,
            role=args.role
        )
        if created:
            print(f"SUCCESS: Admin account '{username}' created successfully with role '{args.role}'.")
        else:
            print("Error: Failed to create user in database.")
            sys.exit(1)

if __name__ == "__main__":
    main()
