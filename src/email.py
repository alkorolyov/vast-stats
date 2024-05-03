import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging



def send_error_email(subject, body):
    # Email configuration
    sender_email = 'ergot.notification@gmail.com'
    receiver_email = 'korol.al@gmail.com'
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587  # Port for TLS

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject

    # Add body to email
    message.attach(MIMEText(body, 'plain'))

    # Create SMTP session for sending the mail
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        server.login(sender_email, 'czkg kwcw wtwq cdnb')  # Replace 'your_password' with your email password

        # Send the email
        server.sendmail(sender_email, receiver_email, message.as_string())
        logging.info("Error email sent successfully!")

    except Exception as e:
        logging.warning(f"Error sending email: {e}")

    finally:
        # Close the SMTP server
        server.quit()


# # TODO remove in production
# def send_error_email(subject, body):
#     logging.info("Email should be sent!")
#     pass
