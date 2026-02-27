import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create sample performance chart
def create_performance_chart():
    agents = ['ReAct-GPT4', 'AutoGPT', 'LangChain', 'CrewAI', 'BabyAGI']
    accuracy = [0.87, 0.78, 0.82, 0.85, 0.76]
    speed = [1200, 2100, 950, 1800, 2400]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy chart
    bars1 = ax1.bar(agents, accuracy, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
    ax1.set_title('AI Agent Accuracy Comparison')
    ax1.set_ylabel('Accuracy Score')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Speed chart
    bars2 = ax2.bar(agents, speed, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
    ax2.set_title('AI Agent Response Time Comparison')
    ax2.set_ylabel('Response Time (ms)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height)}ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('data/multimodal_agents/images/agent_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create architecture diagram
def create_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Define components
    components = {
        'User Query': (2, 7, '#FFE5B4'),
        'Query Engine': (2, 6, '#B4E5FF'),
        'Retriever': (1, 5, '#C8FFB4'),
        'Vector Store': (0.5, 4, '#FFB4B4'),
        'LLM': (3, 5, '#E5B4FF'),
        'Documents': (0.5, 3, '#FFCCB4'),
        'Response': (2, 2, '#B4FFE5')
    }
    
    # Draw components
    for name, (x, y, color) in components.items():
        rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6, 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrows = [
        ((2, 6.7), (2, 6.3)),  # User Query -> Query Engine
        ((2, 5.7), (1.4, 5.3)),  # Query Engine -> Retriever
        ((2, 5.7), (2.6, 5.3)),  # Query Engine -> LLM
        ((1, 4.7), (0.9, 4.3)),  # Retriever -> Vector Store
        ((0.5, 3.7), (0.7, 3.3)),  # Vector Store -> Documents
        ((2.5, 4.7), (2.3, 2.3)),  # LLM -> Response
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(1, 8)
    ax.set_title('RAG Agent Architecture', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/multimodal_agents/images/rag_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create agent types visualization
def create_agent_types_chart():
    # Agent types and their characteristics
    agent_types = ['Reasoning\nAgents', 'Tool-Using\nAgents', 'Autonomous\nAgents', 
                  'Multi-Agent\nSystems', 'Conversational\nAgents']
    complexity = [3, 2, 5, 4, 2]
    popularity = [4, 5, 3, 3, 4]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(agent_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, complexity, width, label='Complexity', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, popularity, width, label='Popularity', color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Agent Types')
    ax.set_ylabel('Score (1-5)')
    ax.set_title('AI Agent Types: Complexity vs Popularity')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_types)
    ax.legend()
    ax.set_ylim(0, 6)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('data/multimodal_agents/images/agent_types_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Creating sample images...")
    create_performance_chart()
    create_architecture_diagram()
    create_agent_types_chart()
    print("Sample images created successfully!")
