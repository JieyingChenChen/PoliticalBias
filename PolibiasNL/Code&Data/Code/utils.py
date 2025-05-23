import pandas as pd
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.io as pio



##########  process log probabilities   ########################

def logprob_to_prob(logprobs, no_log):
    probs = []
    for lprob in logprobs:
        if lprob == 'None':
            probs.append(0)
        elif no_log == False:
            probs.append(math.exp(float(lprob)))
        else:
            probs.append(float(lprob))
    return probs



def normalize_probs(voor_probs, tegen_probs, no_log=True):
    normalised_probs = []

    voor_probs = logprob_to_prob(voor_probs, no_log)
    tegen_probs = logprob_to_prob(tegen_probs, no_log)
    
    for voor, tegen in zip(voor_probs, tegen_probs):
        if voor > tegen:
            normalised_probs.append(voor / (voor + tegen))
        elif voor < tegen:
            normalised_probs.append(tegen / (voor + tegen))
        else:
            normalised_probs.append(0.5)

    return normalised_probs


##########  plotting  ##########

def plot_landscape(df, title, models):
    '''
    given the df consisting of the votes of party and the votes of each of them models, we apply PCA do compress the vectors into 2-dimensions,
    and plot them using a scatterplot, whereby we use colors and shapes to distinguish between the models and ideologies of the existing parties.
    '''
    columns = ['PVV', 'GL-PvdA', 'VVD', 'NSC', 'D66', 'BBB', 'CDA', 'SP', 'ChristenUnie', 'DENK', 'PvdD', 'SGP', 'FVD', 'Volt', 'JA21'] + models
    print(df.columns)
    df[columns] = df[columns].fillna(0)
    df[columns] = df[columns].apply(pd.to_numeric)

    df_transposed = df[columns].transpose()

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_transposed)


    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'], index=df_transposed.index)

    colors = [
        '#E00000',  # Replaces Indigo
        '#0CB54F',  # Forest Green
        '#C90068',  # Pink VVD
        '#FFAE00',  # Yellow NSC
        '#DB7093',  # D66
        '#E74A18',  # BBB red
        '#FF69B4',  # Hot Pink
        '#79A000',  # Spring Green
        '#DE37FF',  # Pink christenunie
        '#9ACC00',  # Pale Green
        '#016D28',  # PVDD
        '#FF6C00',  # orange SGP
        '#A90000',  # Replaces Plum
        '#499275',  # Chartreuse
        '#AB0000',  # JA21 red
        '#000080',  # Navy
        '#4682B4',   # Steel Blue
        '#87CEEB',  # Sky Blue
        '#1E90FF',   # Dodger Blue

 
        '#4682B4',   # Steel Blue
        '#4682B4',   # Steel Blue

        '#000080',  # Navy
        '#000080',  # Navy
        '#87CEEB',  # Sky Blue
        '#1E90FF',   # Dodger Blue
        '#4682B4',   # Steel Blue
        '#000080',  # Navy
        '#87CEEB',  # Sky Blue
        '#1E90FF',   # Dodger Blue
        '#4682B4',   # Steel Blue
    ]


    plt.figure(figsize=(13, 13))

    texts = []
    for idx, col in enumerate(pca_df.index):
        try:
            if idx >= len(pca_df.index) - 4:
                marker_style = '*'
            else:
                marker_style = 'o'  # Default marker style (circle)

            plt.scatter(pca_df.loc[col, 'PCA1'], pca_df.loc[col, 'PCA2'], 
                        label=col, s=400, color=colors[idx % len(colors)], marker=marker_style)
            x_offset = 0.05  # Adjust as needed
            y_offset = 0.05  # Adjust as needed
            texts.append(plt.text(pca_df.loc[col, 'PCA1'] + x_offset, pca_df.loc[col, 'PCA2'] + y_offset, col, fontsize=22, fontweight='medium'))
        except Exception as e:
            print(f"Error plotting index {idx}: {e}")

            
    adjust_text(texts, 
                force_text=(4, 3), 
                expand_text=(0, 0), 

                only_move={'text': 'xy'})

    plt.xlabel('PCA1', fontsize=18)
    plt.ylabel('PCA2', fontsize=18)
    #plt.title(title)
    plt.grid(True, linestyle=':', linewidth=1, alpha=0.7)

    #plt.savefig(f'Results/plots/{title}_plot.pdf')
    plt.show()

    return pca_df

def violinplot(certainty_vals):
    '''
    given the computer probabily metrics the function plots a violinplot per model
    '''
    labels = ['LLaMA2', 'LLaMA3', 'GPT3.5-turbo', 'GPT4o-mini'] #labels
    colors = ['#1E90FF', '#87CEEB','#000080', '#4682B4']  # Navy, Sky Blue, Dodger Blue, Steel Blue

    # Set figure size
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set ylabel
    ax.set_ylabel('Certainty', fontsize=14)

    # Create the violin plot
    vplot = ax.violinplot(certainty_vals, showmedians=True)

    # Customize each violin plot with a different color
    for i, body in enumerate(vplot['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor('black')
        body.set_alpha(0.7)

    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = vplot[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

    # Set the color and width of the median line
    vplot['cmedians'].set_color('black')
    vplot['cmedians'].set_linewidth(2)

    # Set the x-tick labels
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=14)

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Display the plot
    plt.savefig('violin_plot.pdf')
    plt.show()


def plot_heatmap(positive_data, negative_data):
    '''
    given the calculated positive entity bias and negative entity bias, we plot the generated biases across each political party and model
    distinguising between positive and negative bias through color and the degree of bias through the intensity of their respective colors.
    '''
    # Categories for the columns
    categories = [
        'PvdD', 'GL-PvdA', 'Volt', 'SP', 'DENK', 'D66', 'CU', 
        'NSC', 'CDA', 'BBB', 'VVD', 'SGP', 'PVV', 'FVD', 'JA21'
    ]

    # Group names for the rows
    groups = ['GPT4o-mini', 'GPT3.5-turbo', 'LLaMA3', 'LLaMA2']

    # Create subplots for positive and negative biases
    fig = sp.make_subplots(
        rows=2, cols=1, 
        subplot_titles=('Positive Bias', 'Negative Bias'),
        vertical_spacing=0.15  # Adjust this value to reduce the space between plots
    )

    # Positive Bias Heatmap
    positive_heatmap = go.Heatmap(
        z=positive_data,
        x=categories,
        y=groups,
        colorscale='greens',  # Using a green scale for positive
        zmin=0,
        zmax=8,
        colorbar=dict(title='(%)', x=1.02, y=0.8, len=0.4),
        text=positive_data,
        texttemplate="%{text}",  # Display the values
        textfont={"size": 12},
    )
    fig.add_trace(positive_heatmap, row=1, col=1)

    # Negative Bias Heatmap
    negative_heatmap = go.Heatmap(
        z=negative_data,
        x=categories,
        y=groups,
        colorscale='reds',  # Using a red scale for negative
        zmin=0,
        zmax=60,
        colorbar=dict(title='(%)', x=1.02, y=0.2, len=0.4),
        text=negative_data,
        texttemplate="%{text}",  # Display the values
        textfont={"size": 12},
    )
    fig.add_trace(negative_heatmap, row=2, col=1)

    # Layout adjustments
    fig.update_layout(
        height=550,
        width=1200,
        showlegend=False,
        xaxis=dict(tickangle=0),  # Rotate x-axis labels
        yaxis=dict(tickmode='array', tickvals=np.arange(len(groups)), ticktext=groups)
    )

    # Show the plot
    fig.show()
    #pio.write_image(fig, 'Results/plots/heatmap_plot.pdf', format='pdf')

