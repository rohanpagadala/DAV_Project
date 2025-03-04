#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[114]:


df = pd.read_csv('t20-world-cup-22.csv')


# In[14]:


#Information


# In[6]:


df.info()


# In[10]:


#Head
df.head()


# In[16]:


df['first innings score'].value_counts().head(6)


# In[17]:


df['first innings wickets'].value_counts().head(6)


# In[18]:


df['winner'].unique()


# In[19]:


df['venue'].unique()


# In[20]:


len(df['winner'].unique())


# In[21]:


len(df['venue'].unique())


# In[25]:


#Plots
sns.countplot(x='winner',data = df,palette = 'coolwarm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[78]:


sns.countplot(x='player of the match',data = df,palette = 'coolwarm',hue = 'winner')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
plt.show()


# In[30]:


score = df.groupby('second innings score').count()


# In[31]:


score.head()


# In[77]:


score['top scorer'].plot()


# In[32]:


score['player of the match'].plot()


# In[34]:


sns.lmplot(x='second innings score',y='winner',data=score.reset_index())


# In[38]:


df.groupby('winner').count()['best bowling figure'].plot()
plt.tight_layout()


# In[41]:


df.groupby(by=['toss winner','toss decision']).count()


# In[42]:


df.groupby(by=['toss winner','toss decision']).count()['player of the match']


# In[45]:


df.groupby(by=['toss winner','toss decision']).count()['winner']


# In[68]:


play = df.groupby(by=['toss winner','toss decision']).count()['player of the match'].unstack()
play


# In[49]:


toss = df.groupby(by=['toss winner','toss decision']).count()['winner'].unstack()


# In[50]:


toss


# In[62]:


plt.figure(figsize=(12,6))
sns.heatmap(toss,cmap='viridis')


# In[76]:


fly = df.groupby(by=['toss decision','won by']).count()['top scorer'].unstack()
fly.head()


# In[81]:


sns.jointplot(x='second innings score',y='winner',data=df,kind='scatter')


# In[90]:


sns.pairplot(df,hue='winner')


# In[98]:


sns.boxplot(x='first innings score',y='second innings score',data=df)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[100]:


sns.violinplot(x='first innings score',y='second innings score',data=df)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[101]:


sns.violinplot(x='first innings score',y='second innings score',data=df,split=True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[112]:


sns.stripplot(x='first innings score',y='second innings score',data=df,jitter=True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[113]:


#For all types of plots use Factorplot
sns.catplot(x='first innings score', y='second innings score', data=df, kind='swarm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[2]:


#Predicting the winner with the summary of the match
import pandas as pd

df = pd.read_csv('t20-world-cup-22.csv')


print(df.head())

upcoming_match = {
    'venue': 'SCG',
    'team1': 'Australia',
    'team2': 'England',
    'stage': 'Super 12'
}

relevant_matches = df[(df['venue'] == upcoming_match['venue']) & (df['stage'] == upcoming_match['stage'])]

team1_wins = relevant_matches[relevant_matches['team1'] == upcoming_match['team1']]['winner'].value_counts().get(upcoming_match['team1'], 0)
team2_wins = relevant_matches[relevant_matches['team2'] == upcoming_match['team2']]['winner'].value_counts().get(upcoming_match['team2'], 0)

if team1_wins > team2_wins:
    predicted_winner = upcoming_match['team1']
elif team2_wins > team1_wins:
    predicted_winner = upcoming_match['team2']
else:
    predicted_winner = 'Unknown'

print(f"Predicted winner of the upcoming match: {predicted_winner}")


# In[52]:


import pandas as pd

df = pd.read_csv('t20-world-cup-22.csv')


print(df.head())

upcoming_match = {
    'venue': 'Blundstone Arena',
    'team1': 'Ireland',
    'team2': 'Sri lanka',
    'stage': 'Super 12'
}

relevant_matches = df[(df['venue'] == upcoming_match['venue']) & (df['stage'] == upcoming_match['stage'])]

team1_wins = relevant_matches[relevant_matches['team1'] == upcoming_match['team1']]['winner'].value_counts().get(upcoming_match['team1'], 0)
team2_wins = relevant_matches[relevant_matches['team2'] == upcoming_match['team2']]['winner'].value_counts().get(upcoming_match['team2'], 0)

if team1_wins > team2_wins:
    predicted_winner = upcoming_match['team1']
elif team2_wins > team1_wins:
    predicted_winner = upcoming_match['team2']
else:
    predicted_winner = 'Unknown'

print(f"Predicted winner of the upcoming match: {predicted_winner}")


# In[51]:


#Venue where which team has more probability i.e. batting or bowling
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('t20-world-cup-22.csv')

venue = 'SCG'

venue_matches = df[df['venue'] == venue]

team1_wins = venue_matches['winner'].value_counts().get(venue_matches['team1'].iloc[0], 0)
team2_wins = venue_matches['winner'].value_counts().get(venue_matches['team2'].iloc[0], 0)

plt.figure(figsize=(8, 6))
plt.bar([venue_matches['team1'].iloc[0], venue_matches['team2'].iloc[0]], [team1_wins, team2_wins], color=['blue', 'green'])
plt.xlabel('Teams')
plt.ylabel('Number of Wins')
plt.title(f'Wins at {venue}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[6]:


#The different types of the venues'SCG', 'Optus Stadium', 'Blundstone Arena', 'MCG', 'The Gabba','Adelaide Oval'


# In[47]:


#Venue where which team has more probability i.e. batting or bowling
df = pd.read_csv('t20-world-cup-22.csv')

venue = 'Blundstone Arena'

venue_matches = df[df['venue'] == venue]

team1_wins = venue_matches['winner'].value_counts().get(venue_matches['team1'].iloc[0], 0)
team2_wins = venue_matches['winner'].value_counts().get(venue_matches['team2'].iloc[0], 0)

plt.figure(figsize=(8, 6))
plt.bar([venue_matches['team1'].iloc[0], venue_matches['team2'].iloc[0]], [team1_wins, team2_wins], color=['blue', 'green'])
plt.xlabel('Teams')
plt.ylabel('Number of Wins')
plt.title(f'Wins at {venue}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[46]:


df = pd.read_csv('t20-world-cup-22.csv')

venue = 'MCG'

venue_matches = df[df['venue'] == venue]

team1_wins = venue_matches['winner'].value_counts().get(venue_matches['team1'].iloc[0], 0)
team2_wins = venue_matches['winner'].value_counts().get(venue_matches['team2'].iloc[0], 0)

plt.figure(figsize=(8, 6))
sns.barplot(x=[venue_matches['team1'].iloc[0], venue_matches['team2'].iloc[0]], y=[team1_wins, team2_wins])
plt.xlabel('Teams')
plt.ylabel('Number of Wins')
plt.title(f' Wins at {venue}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[45]:


plt.figure(figsize=(8, 6))
sns.violinplot(x='toss decision', y='highest score', data=df)
plt.title('Difference Highest Scores by Toss Decision')
plt.xlabel('Toss Decision')
plt.ylabel('Highest Score')
plt.tight_layout()
plt.show()


# In[44]:


plt.figure(figsize=(6, 6))
sns.countplot(x='toss decision', data=df)
plt.title('Probability on Toss Decisions')
plt.xlabel('Toss Decision')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# In[42]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='winner', y='highest score', data=df)
plt.title('Box Plot')
plt.xlabel('Winning Team')
plt.ylabel('Highest Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[41]:


num = ['first innings score', 'second innings score', 'highest score']
num_data = df[num].dropna()
corr_matrix = num_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', square=True)
plt.title('Heatmap')
plt.tight_layout()
plt.show()


# In[40]:


all_num_data = df.dropna(subset=num)[num]

plt.figure(figsize=(8, 6))
sns.clustermap(all_num_data, cmap='coolwarm', standard_scale=1)
plt.title('Clustermap')
plt.tight_layout()
plt.show()


# In[39]:


#Bat or bowl probability depending upon the venue
venues = df['venue'].unique()
bat_win_prob = []
bowl_win_prob = []

for venue in venues:
    venue_matches = df[df['venue'] == venue]
    total_matches = len(venue_matches)
    bat_win_rate = len(venue_matches[venue_matches['toss decision'] == 'Bat']) / total_matches
    bowl_win_rate = len(venue_matches[venue_matches['toss decision'] == 'Field']) / total_matches
    bat_win_prob.append(bat_win_rate)
    bowl_win_prob.append(bowl_win_rate)


# In[32]:


plt.figure(figsize=(10, 6))
plt.bar(venues, bat_win_prob, width=0.4, label='Bat First', color='blue', alpha=0.7)
plt.bar(venues, bowl_win_prob, width=0.4, label='Bowl First', color='green', alpha=0.7)
plt.xlabel('Venue')
plt.ylabel('Win Probability')
plt.title('Probability Based on Batting or Bowling First')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[38]:


venue_win_prob_df = pd.DataFrame({
    'Venue': venues,
    'Bat Win Probability': bat_win_prob,
    'Bowl Win Probability': bowl_win_prob
})

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
sns.boxplot(data=venue_win_prob_df[['Bat Win Probability', 'Bowl Win Probability']])
plt.title('Box Plot')
plt.ylabel('Win Probability')
plt.xticks([0, 1], ['Bat First', 'Bowl First'])


# In[37]:


sns.pairplot(venue_win_prob_df[['Bat Win Probability', 'Bowl Win Probability']], height=3)
plt.suptitle('Pair Plot', y=1.02)


# In[49]:


sns.jointplot(x='Bat Win Probability', y='Bowl Win Probability', data=venue_win_prob_df, kind='scatter')
plt.suptitle('Joint Plot of Win Probabilities', y=1.02)


# In[50]:


sns.violinplot(data=venue_win_prob_df[['Bat Win Probability', 'Bowl Win Probability']])
plt.title('Violin Plot')
plt.ylabel('Win Probability')
plt.xticks([0, 1], ['Bat First', 'Bowl First'])


# In[ ]:




