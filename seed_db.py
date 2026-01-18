import os
import django
import uuid
from datetime import datetime

# Setup Django - override MONGO_URI for local development
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.settings')
os.environ['MONGO_URI'] = 'mongodb://localhost:27017/vantage'  # Use localhost instead of mongodb

django.setup()

from app.models import User, GeminiConversation, ConversationMessage

def seed_database():
    print("Starting database seeding...")
    
    try:
        # Create a single user
        user_data = {'name': 'Demo User', 'email': 'demo@example.com', 'auth0_id': 'auth0|demo123'}
        
        try:
            user = User.objects.get(email=user_data['email'])
            print(f"Found user: {user.email}")
        except User.DoesNotExist:
            user = User(**user_data)
            user.save()
            print(f"Created user: {user.email}")
        
        # Create multiple conversation sessions for the user
        conversation_topics = [
            {
                'title': 'Introduction to Machine Learning',
                'messages': [
                    {'role': 'user', 'content': 'What is machine learning?'},
                    {'role': 'assistant', 'content': 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.'},
                    {'role': 'user', 'content': 'Can you give me an example?'},
                    {'role': 'assistant', 'content': 'Sure! Email spam filters are a great example. They learn to distinguish spam from legitimate emails based on patterns in data.'},
                ]
            },
            {
                'title': 'Understanding Neural Networks',
                'messages': [
                    {'role': 'user', 'content': 'How do neural networks work?'},
                    {'role': 'assistant', 'content': 'Neural networks are inspired by the human brain. They consist of interconnected nodes (neurons) that process information through layers.'},
                    {'role': 'user', 'content': 'What are the main types?'},
                    {'role': 'assistant', 'content': 'The main types include Convolutional Neural Networks (CNN) for image processing, Recurrent Neural Networks (RNN) for sequences, and Transformers for NLP tasks.'},
                ]
            },
            {
                'title': 'Python for Data Science',
                'messages': [
                    {'role': 'user', 'content': 'What Python libraries should I learn for data science?'},
                    {'role': 'assistant', 'content': 'The essential libraries are: NumPy for numerical computing, Pandas for data manipulation, Scikit-learn for machine learning, and Matplotlib for visualization.'},
                    {'role': 'user', 'content': 'Which one should I start with?'},
                    {'role': 'assistant', 'content': 'I recommend starting with Pandas to learn data manipulation, then moving to NumPy for mathematical operations, and finally Scikit-learn for machine learning algorithms.'},
                ]
            },
            {
                'title': 'Deep Learning Applications',
                'messages': [
                    {'role': 'user', 'content': 'What are some real-world applications of deep learning?'},
                    {'role': 'assistant', 'content': 'Deep learning powers many applications: computer vision for object detection, natural language processing for translation, autonomous vehicles, medical image analysis, and recommendation systems.'},
                ]
            },
            {
                'title': 'Getting Started with TensorFlow',
                'messages': [
                    {'role': 'user', 'content': 'How do I get started with TensorFlow?'},
                    {'role': 'assistant', 'content': 'Start by installing TensorFlow via pip, then explore the official tutorials. Begin with simple models like linear regression before moving to neural networks.'},
                ]
            },
        ]
        
        for i, topic in enumerate(conversation_topics):
            conv = GeminiConversation(
                user_id=str(user.id),
                session_id=str(uuid.uuid4()),
                title=topic['title'],
                total_messages=str(len(topic['messages'])),
                total_images='0',
                total_audio_chunks='0'
            )
            
            # Add messages to conversation
            for msg in topic['messages']:
                conv.messages.append(
                    ConversationMessage(
                        role=msg['role'],
                        message_type='text',
                        content=msg['content']
                    )
                )
            
            conv.save()
            print(f"Created conversation {i+1}: {topic['title']} ({len(topic['messages'])} messages)")
        
        print(f"\nSeeding completed! Created 5 conversations for user: {user.email}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    seed_database()