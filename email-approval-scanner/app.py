import streamlit as st
import imaplib
import email
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import pickle
import os
import urllib.parse
import json
import tempfile

# Set page config ONCE at the very top
st.set_page_config(
    page_title="STRICT AI Email Approval Scanner",
    page_icon="🤖",
    layout="wide"
)

# Machine Learning imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
except ImportError as e:
    st.error(f"ML libraries missing: {e}")
    st.stop()

# Gmail API imports for real message IDs
GMAIL_API_AVAILABLE = False
try:
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    import base64
    GMAIL_API_AVAILABLE = True
    print("✅ Gmail API libraries available - Direct email links enabled")
except ImportError:
    GMAIL_API_AVAILABLE = False
    print("❌ Gmail API libraries not available - Using IMAP fallback")

class StrictApprovalAIModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
        self.training_emails = [
            ("With reference to the below mail, all GADs and Data sheets are ok. Kindly submit the QAP for the same.", 1),
            ("Dear Sir, Approved drawing is enclosed, Kindly arrange to share the QAP also.", 1),
            ("Dear sir, These are the approval of documents for single girder EOT crane. Please review this and start with manufacturing.", 1),
            ("Please find attached Approved Drg & kindly start Manufacturing.", 1),
            ("Please find enclosed herewith Approved Datasheet with GA Drawing. Kindly proceed for dispatch", 1),
            ("We have gone through the GAD and it is approved from our end.", 1),
            ("Dear Team, PFA approved Cat-I document. Kindly start the manufacturing and share your manufacturing & inspection schedule.", 1),
            ("Please acknowledge the same Drawing and QAP is approved, please proceed for manufacturing.", 1),
            ("PFA the GA & data sheet approved. Thanks & Regards", 1),
            ("We hereby approve the attached drawing. You are requested to supply as per the revised GA", 1),
            ("Please get the drawing approved from your side and submit for our review", 0),
            ("Do the needful for drg approval from your technical team", 0),
            ("We require approval from you before we can proceed with this project", 0),
            ("Please revise the drawing as per comments and resubmit for approval", 0),
            ("Kindly get internal approval and then submit the documents", 0),
            ("Please submit the drawing for our approval", 0),
            ("Drawing approval is required from your end", 0),
            ("We are waiting for your approval on the submitted documents", 0)
        ]
    
    def preprocess_text(self, text):
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r'from:.*?to:.*?\n', '', text, flags=re.DOTALL)
        text = re.sub(r'sent:.*?\n', '', text)
        text = re.sub(r'subject:.*?\n', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\+?[\d\s\-\(\)]{10,}', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_strict_features(self, text):
        text_lower = text.lower()
        
        positive_features = {
            'has_approved_document': int(bool(re.search(r'approved.{0,20}(drawing|document|ga|datasheet|drg)', text_lower))),
            'has_document_approved': int(bool(re.search(r'(drawing|document|ga|datasheet|drg).{0,20}approved', text_lower))),
            'has_approved_attached': int('approved' in text_lower and ('attached' in text_lower or 'enclosed' in text_lower)),
            'has_pfa_approved': int('pfa' in text_lower and 'approved' in text_lower),
            'has_find_approved': int('find' in text_lower and 'approved' in text_lower),
            'has_start_manufacturing': int('start' in text_lower and 'manufacturing' in text_lower),
            'has_proceed_manufacturing': int('proceed' in text_lower and 'manufacturing' in text_lower),
            'has_go_ahead': int('go ahead' in text_lower),
            'has_formal_approval': int('formal approval' in text_lower),
            'has_hereby_approve': int('hereby approve' in text_lower),
            'has_approved_from_end': int('approved from our end' in text_lower),
            'has_mfc': int('mfc' in text_lower),
            'has_signed_stamped': int('signed' in text_lower and 'stamp' in text_lower),
        }
        
        negative_features = {
            'requests_approval': int(bool(re.search(r'(please|kindly).{0,30}(get|provide|arrange).{0,20}approval', text_lower))),
            'needs_approval': int(bool(re.search(r'(need|require|want).{0,20}approval', text_lower))),
            'approval_from_you': int('approval from you' in text_lower or 'your approval' in text_lower),
            'get_approved': int('get' in text_lower and 'approved' in text_lower and 'from' in text_lower),
            'submit_for_approval': int('submit' in text_lower and 'approval' in text_lower),
            'approval_required': int('approval required' in text_lower),
            'approval_pending': int('approval pending' in text_lower),
        }
        
        all_features = {**positive_features, **negative_features}
        all_features.update({
            'word_count': len(text.split()),
            'has_question_mark': int('?' in text),
            'has_please': int('please' in text_lower),
        })
        
        return all_features
    
    def train_model(self):
        texts = []
        labels = []
        
        for text, label in self.training_emails:
            processed_text = self.preprocess_text(text)
            texts.append(processed_text)
            labels.append(label)
        
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        X_text = self.vectorizer.fit_transform(texts)
        
        additional_features = []
        for text in texts:
            features = self.extract_strict_features(text)
            additional_features.append(list(features.values()))
        
        X_additional = np.array(additional_features)
        X_combined = np.hstack([X_text.toarray(), X_additional])
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight={0: 1, 1: 2}
        )
        
        self.model.fit(X_combined, labels)
        self.is_trained = True
        return accuracy_score(labels, self.model.predict(X_combined))
    
    def predict_approval(self, email_text, subject="", sender=""):
        if not self.is_trained:
            self.train_model()
        
        full_text = f"{subject} {email_text}"
        processed_text = self.preprocess_text(full_text)
        
        # Ultra strict rejection patterns
        ultra_strict_rejection_patterns = [
            r'please.{0,50}get.{0,20}approval',
            r'kindly.{0,50}get.{0,20}approval',
            r'need.{0,30}approval.{0,30}from',
            r'require.{0,30}approval.{0,30}from',
            r'approval.{0,30}from.{0,30}you',
            r'submit.{0,30}for.{0,20}approval',
            r'approval.{0,20}required',
            r'approval.{0,20}pending',
        ]
        
        for pattern in ultra_strict_rejection_patterns:
            if re.search(pattern, processed_text):
                return {
                    'is_approved': False,
                    'confidence': 98.0,
                    'probability_approved': 2.0,
                    'reasoning': [f'Ultra strict rejection: {pattern[:30]}'],
                    'features_detected': ['ultra_strict_rejection_pattern']
                }
        
        # Must have explicit approval language
        must_have_approval_patterns = [
            r'approved.{0,20}(drawing|document|ga|datasheet|drg)',
            r'(drawing|document|ga|datasheet|drg).{0,20}approved',
            r'hereby\s+approve',
            r'formal\s+approval',
            r'approved\s+from\s+our\s+end',
            r'pfa.{0,20}approved',
        ]
        
        has_explicit_approval = any(re.search(pattern, processed_text) for pattern in must_have_approval_patterns)
        
        if not has_explicit_approval:
            return {
                'is_approved': False,
                'confidence': 85.0,
                'probability_approved': 15.0,
                'reasoning': ['No explicit approval language'],
                'features_detected': ['no_explicit_approval']
            }
        
        # ML prediction
        X_text = self.vectorizer.transform([processed_text])
        features = self.extract_strict_features(processed_text)
        X_additional = np.array([list(features.values())])
        X_combined = np.hstack([X_text.toarray(), X_additional])
        
        probabilities = self.model.predict_proba(X_combined)[0]
        approval_prob = probabilities[1] * 100
        
        ULTRA_STRICT_THRESHOLD = 0.85
        is_approved = probabilities[1] > ULTRA_STRICT_THRESHOLD
        
        if is_approved:
            strong_positives = [
                features['has_approved_document'],
                features['has_document_approved'],
                features['has_start_manufacturing'],
                features['has_proceed_manufacturing'],
                features['has_go_ahead'],
                features['has_formal_approval'],
                features['has_hereby_approve'],
                features['has_approved_attached'],
                features['has_pfa_approved'],
                features['has_find_approved']
            ]
            
            if sum(strong_positives) < 2:
                is_approved = False
                approval_prob = min(approval_prob, 35.0)
        
        confidence = max(probabilities) * 100
        
        return {
            'is_approved': is_approved,
            'confidence': confidence,
            'probability_approved': approval_prob,
            'reasoning': [f"ML confidence: {confidence:.1f}%"],
            'features_detected': [k for k, v in features.items() if v > 0]
        }

class GmailAPIScanner:
    """Gmail API scanner that uses uploaded credentials"""
    
    def __init__(self, email, password, credentials_data):
        self.email = email
        self.password = password
        self.credentials_data = credentials_data
        self.service = None
        self.ai_model = StrictApprovalAIModel()
        self.scopes = ['https://www.googleapis.com/auth/gmail.readonly']
    
    def authenticate_gmail_api(self):
        """Authenticate with Gmail API using uploaded credentials"""
        try:
            # Create temporary file with credentials data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(self.credentials_data, temp_file)
                temp_credentials_path = temp_file.name
            
            creds = None
            
            # Check if token exists in session state
            if 'gmail_token' in st.session_state:
                try:
                    creds = Credentials.from_authorized_user_info(st.session_state.gmail_token, self.scopes)
                except:
                    creds = None
            
            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        temp_credentials_path, self.scopes)
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials to session state
                st.session_state.gmail_token = json.loads(creds.to_json())
            
            # Clean up temp file
            os.unlink(temp_credentials_path)
            
            self.service = build('gmail', 'v1', credentials=creds)
            return True
            
        except Exception as e:
            st.error(f"Gmail API authentication failed: {e}")
            return False
    
    def scan_recent_emails_api(self, days=7, max_emails=None, company_domains=None):
        """Scan emails using Gmail API - provides real Gmail message IDs"""
        if company_domains is None:
            company_domains = ['revacranes.com', 'eiepl.in']
            
        try:
            if not self.service:
                if not self.authenticate_gmail_api():
                    return [], []
            
            # Calculate date range
            since_date = datetime.now() - timedelta(days=days)
            query = f'after:{since_date.strftime("%Y/%m/%d")}'
            
            # Get messages
            results = self.service.users().messages().list(
                userId='me', q=query, maxResults=max_emails
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return [], []
            
            approved_emails = []
            all_analyzed = []
            
            total_messages = len(messages)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, message in enumerate(messages):
                progress_bar.progress((i + 1) / total_messages)
                status_text.text(f"Gmail API analyzing email {i+1}/{total_messages}")
                
                try:
                    # Get Gmail message ID (REAL Gmail message ID!)
                    gmail_msg_id = message['id']
                    
                    msg_detail = self.service.users().messages().get(
                        userId='me', id=gmail_msg_id
                    ).execute()
                    
                    # Extract headers
                    headers = {h['name']: h['value'] for h in msg_detail['payload'].get('headers', [])}
                    
                    subject = headers.get('Subject', '(No Subject)')
                    sender = headers.get('From', 'Unknown Sender')
                    date = headers.get('Date', 'Unknown Date')
                    
                    # Skip internal emails
                    sender_domain = sender.split('@')[-1].lower() if '@' in sender else ""
                    if any(domain in sender_domain for domain in company_domains):
                        continue
                    
                    # Get email content
                    content = self.get_email_content_api(msg_detail)
                    
                    if not content:
                        continue
                    
                    # AI Analysis
                    ai_result = self.ai_model.predict_approval(content, subject, sender)
                    
                    email_data = {
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'content_preview': content[:500] + "..." if len(content) > 500 else content,
                        'is_approved': ai_result['is_approved'],
                        'confidence': ai_result['confidence'],
                        'probability_approved': ai_result['probability_approved'],
                        'ai_reasoning': ai_result['reasoning'],
                        'features_detected': ai_result['features_detected'],
                        'gmail_message_id': gmail_msg_id,  # REAL Gmail message ID!
                        'email_id': gmail_msg_id
                    }
                    
                    all_analyzed.append(email_data)
                    
                    if ai_result['is_approved']:
                        approved_emails.append(email_data)
                
                except Exception as e:
                    continue
            
            progress_bar.empty()
            status_text.empty()
            return approved_emails, all_analyzed
            
        except Exception as e:
            st.error(f"Gmail API error: {e}")
            return [], []
    
    def get_email_content_api(self, msg_detail):
        """Extract email content from Gmail API response"""
        try:
            payload = msg_detail['payload']
            
            def extract_text(payload):
                if 'parts' in payload:
                    for part in payload['parts']:
                        if part['mimeType'] == 'text/plain':
                            data = part['body']['data']
                            return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                        elif 'parts' in part:
                            return extract_text(part)
                elif payload['mimeType'] == 'text/plain':
                    data = payload['body']['data']
                    return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                return ""
            
            content = extract_text(payload)
            return self.extract_latest_email_only(content)
            
        except Exception as e:
            return ""
    
    def extract_latest_email_only(self, email_content):
        """Extract only the latest email content, removing forwarded/replied content"""
        if not email_content:
            return ""
        
        reply_patterns = [
            r'-----Original Message-----',
            r'From:.*?Sent:.*?To:.*?Subject:',
            r'On.*?wrote:',
            r'> ', r'>>',
            r'From: .*?@.*?\n',
            r'________________________________',
        ]
        
        lines = email_content.split('\n')
        latest_content_lines = []
        
        for line in lines:
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in reply_patterns):
                break
            if not latest_content_lines and not line.strip():
                continue
            latest_content_lines.append(line)
        
        return '\n'.join(latest_content_lines).strip()
    
    def close_connection(self):
        """Gmail API doesn't need explicit connection closing"""
        pass

def generate_gmail_link(email_data, gmail_account_index, user_email):
    """Generate Gmail link using the proper message ID"""
    try:
        # Use the Gmail message ID we extracted
        message_id = email_data.get('gmail_message_id', '')
        
        if not message_id:
            # Fallback to email_id if gmail_message_id is not available
            email_id = email_data.get('email_id', '')
            if email_id.isdigit():
                message_id = format(int(email_id), 'x')
            else:
                message_id = email_id
        
        # Use the exact format that was working before
        gmail_url = f"https://mail.google.com/mail/u/{gmail_account_index}/?authuser={user_email}#all/{message_id}"
        return gmail_url
        
    except Exception as e:
        # Fallback to inbox if anything goes wrong
        return f"https://mail.google.com/mail/u/{gmail_account_index}/?authuser={user_email}#inbox"

def main():
    st.title("🤖 STRICT AI Email Approval Scanner")
    st.caption("AI-powered crane approval detection - Upload your Gmail API credentials to get started!")
    
    # Initialize session state
    if 'gmail_account_index' not in st.session_state:
        st.session_state.gmail_account_index = 2
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Gmail API credentials upload
        st.subheader("Gmail API Setup")
        
        if not GMAIL_API_AVAILABLE:
            st.error("Gmail API libraries not installed. This app requires Gmail API for direct email links.")
            st.info("Install with: pip install google-api-python-client google-auth google-auth-oauthlib")
            return
        
        uploaded_creds = st.file_uploader(
            "Upload Gmail API credentials.json", 
            type="json",
            help="Download from Google Cloud Console > APIs & Services > Credentials"
        )
        
        credentials_data = None
        if uploaded_creds:
            try:
                credentials_data = json.loads(uploaded_creds.read())
                st.success("✅ Gmail API credentials loaded!")
            except Exception as e:
                st.error(f"Error reading credentials file: {e}")
                return
        else:
            st.warning("Please upload Gmail API credentials.json to continue")
            with st.expander("How to get Gmail API credentials"):
                st.markdown("""
                1. Go to [Google Cloud Console](https://console.cloud.google.com/)
                2. Create a new project or select existing one
                3. Enable Gmail API
                4. Go to "Credentials" → "Create Credentials" → "Desktop Application"  
                5. Download the JSON file
                6. Upload it here
                """)
            return
        
        st.subheader("Email Settings")
        user_email = st.text_input("Your Gmail Address:", placeholder="your.email@gmail.com")
        user_password = st.text_input("Gmail App Password:", type="password", help="Not your regular password - create an App Password in Gmail settings")
        
        if not user_email or not user_password:
            st.warning("Please enter your Gmail address and app password")
            return
        
        gmail_account_index = st.selectbox(
            "Gmail Account Position:",
            options=[0, 1, 2, 3, 4],
            index=st.session_state.gmail_account_index,
            help="Which Gmail account slot (0=first account in browser)"
        )
        st.session_state.gmail_account_index = gmail_account_index
        
        company_domains = st.text_area(
            "Company Domains (to exclude):",
            value="revacranes.com\neiepl.in",
            help="Emails from these domains will be skipped"
        ).strip().split('\n')
        
        st.subheader("Scan Settings")
        days_to_scan = st.selectbox("Scan Last:", [1, 3, 7, 14, 30], index=2)
        max_emails = st.selectbox("Email Limit:", ["ALL", 50, 100, 200], index=1)
    
    # Main interface
    st.info("All credentials are processed in your session only and are not stored permanently.")
    
    # Scan button
    if st.button("🎯 Start STRICT AI Scan", type="primary"):
        scanner = GmailAPIScanner(user_email, user_password, credentials_data)
        
        max_emails_value = None if max_emails == "ALL" else int(max_emails)
        
        with st.spinner("STRICT AI analyzing emails using Gmail API..."):
            approved_emails, all_emails = scanner.scan_recent_emails_api(
                days=days_to_scan,
                max_emails=max_emails_value,
                company_domains=company_domains
            )
        
        scanner.close_connection()
        
        st.session_state.approved_emails = approved_emails
        st.session_state.all_emails = all_emails
        st.session_state.user_email = user_email
        
        st.success(f"Found {len(approved_emails)} approvals out of {len(all_emails)} emails using Gmail API with real message IDs!")
    
    # Display results
    if 'approved_emails' in st.session_state:
        tab1, tab2 = st.tabs(["✅ Approvals", "📧 All Emails"])
        
        with tab1:
            st.subheader(f"High Confidence Approvals ({len(st.session_state.approved_emails)})")
            
            for i, email in enumerate(st.session_state.approved_emails):
                with st.expander(f"📧 {email['subject'][:60]}... | {email['confidence']:.1f}%"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**From:** {email['sender']}")
                        st.write(f"**Date:** {email['date']}")
                        st.write(f"**Confidence:** {email['confidence']:.1f}%")
                        st.text_area("Content:", email['content_preview'], height=100, key=f"content_{i}")
                    
                    with col2:
                        gmail_link = generate_gmail_link(
                            email, 
                            st.session_state.gmail_account_index, 
                            st.session_state.get('user_email', user_email)
                        )
                        
                        # Direct clickable link that opens in new tab
                        st.markdown(f"""
                        <a href="{gmail_link}" target="_blank" style="
                            display: inline-block;
                            background-color: #1f77b4;
                            color: white;
                            padding: 10px 20px;
                            text-decoration: none;
                            border-radius: 5px;
                            margin: 5px 0;
                        ">📧 Open Gmail Directly</a>
                        """, unsafe_allow_html=True)
                        
                        if st.button("📋 Copy Link", key=f"copy_{i}"):
                            st.code(gmail_link, language=None)
                            st.success("Link displayed above - copy it!")
                        
                        if st.button("🤖 AI Analysis", key=f"ai_{i}"):
                            st.json({
                                'is_approved': email['is_approved'],
                                'confidence': f"{email['confidence']:.1f}%",
                                'reasoning': email['ai_reasoning'],
                                'features': email['features_detected']
                            })
        
        with tab2:
            st.subheader(f"All Analyzed Emails ({len(st.session_state.all_emails)})")
            
            if st.session_state.all_emails:
                # Show all emails in expandable format
                for i, email in enumerate(st.session_state.all_emails):
                    # Determine status
                    status = "✅ APPROVED" if email['is_approved'] else "❌ Not Approved"
                    confidence = email['confidence']
                    
                    with st.expander(f"{status} | {email['subject'][:50]}... | {confidence:.1f}%"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**From:** {email['sender']}")
                            st.write(f"**Date:** {email['date']}")
                            st.write(f"**AI Decision:** {status}")
                            st.write(f"**Confidence:** {confidence:.1f}%")
                            st.text_area("Content:", email['content_preview'], height=80, key=f"all_content_{i}")
                        
                        with col2:
                            gmail_link = generate_gmail_link(
                                email, 
                                st.session_state.gmail_account_index, 
                                st.session_state.get('user_email', user_email)
                            )
                            
                            # Direct clickable link
                            st.markdown(f"""
                            <a href="{gmail_link}" target="_blank" style="
                                display: inline-block;
                                background-color: #1f77b4;
                                color: white;
                                padding: 8px 16px;
                                text-decoration: none;
                                border-radius: 4px;
                                margin: 4px 0;
                            ">📧 Open Gmail</a>
                            """, unsafe_allow_html=True)
                            
                            if st.button("📋 Copy Link", key=f"copy_all_{i}"):
                                st.code(gmail_link, language=None)
                                st.success("Link displayed!")
                            
                            if st.button("🤖 AI Details", key=f"ai_all_{i}"):
                                st.json({
                                    'is_approved': email['is_approved'],
                                    'confidence': f"{confidence:.1f}%",
                                    'reasoning': email['ai_reasoning'],
                                    'features': email['features_detected']
                                })
                
                # Summary table for overview
                st.subheader("Summary Table")
                df_data = []
                for email in st.session_state.all_emails:
                    sender_short = email['sender'].split('<')[-1].split('>')[0] if '<' in email['sender'] else email['sender'][:30]
                    df_data.append({
                        'Subject': email['subject'][:40] + "..." if len(email['subject']) > 40 else email['subject'],
                        'From': sender_short,
                        'Decision': "✅ APPROVED" if email['is_approved'] else "❌ Not Approved",
                        'Confidence': f"{email['confidence']:.1f}%",
                        'Date': email['date'][:10] if len(email['date']) > 10 else email['date']
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No emails analyzed yet. Run a scan to see results.")

if __name__ == "__main__":
    main()