import argparse
import csv
import os
import pickle
import random
import stopwordsiso as stopwords
import sys
import torch

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from collections import Counter
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from umap import UMAP


# Set the seed for random sampling
random.seed(42)

# Constants (dataset file names, models and specifics)
MAPPINGS_FILENAME = "fact_check_post_mapping.csv"
POSTS_FILENAME = "posts.csv"
FACTCHECKS_FILENAME = "fact_checks.csv"
EMBEDDINGS_FILENAME = "embeddings.pkl"
TOPICS_FILENAME = "topics.pkl"
RES_FOLDERNAME = "res"
LOG_FOLDERNAME = "logs"
# ME5_PREFIX = "query: "
TEMB_MODEL = "intfloat/multilingual-e5-large" # "all-MiniLM-L6-v2"


def create_output_file(data_folder_path, train_neg, strategy, k):
    """
    A function that creates an output file containing negative pairs only
    given a strategy and a negative-to-positive ratio in a format similar 
    to the one of "fact_check_post_mapping.csv". The output file will be
    named as "negatives-$strategy-$k.csv".

    Parameters
    ----------
        data_folder_path: str
            The path to the folder containing the .csv data files and where to write
        train_neg: dict
            A dictionary of (negative) training post-factcheck(s) pairs
        strategy: str
            The strategy applied for sampling negative pairs
        k: int
            The ratio of negatives that has been sampled (compared to the positives)
    """

    # Variables for filename and headerline
    filename = "negatives-" + strategy + "-" + str(k) + ".csv"
    header_line = "post_id,fc_id,relationship,split\n"

    # Print a warning if we are overwriting an already existing file storing negatives
    if os.path.exists(os.path.join(data_folder_path, filename)):
        print(f"WARNING. Overwriting an existing {filename} file...")

    # Create the output file and set the header
    print(f"Writing negative examples to {filename}...")
    output_file = open(os.path.join(data_folder_path, filename), "w")
    output_file.write(header_line)

    # Write the negative examples on the output file and close it
    for p_id, fc_ids in train_neg.items():
        fc_ids_str = ";".join([str(fc_id) for fc_id in fc_ids])
        output_file.write(str(p_id) + "," + fc_ids_str + ",negative,train\n")
    output_file.close()
    print("=> Done.")


def get_negatives_by_topic(
    post_id, curr_cands, n, p_ids_to_texts, fc_ids_to_texts, topics_dict):
    """
    A function that samples n (k*factor) negative fact check ids at random
    by ensuring they belong to the same topic as the input post. The fact checks 
    are taken from a list of candidate fact checks ids for a given post id.

    Parameters
    ----------
        post_id: int
            The id of the post
        curr_cands: list
            The fact check ids of the negative candidates
        n: int
            The number of negatives to sample (i.e., k*factor)
        p_ids_to_texts: dict
            A dictionary of post texts with id as key
        fc_ids_to_texts: dict
            A dictionary of fact check texts with id as key
        topics_dict: dict
            A dictionary storing both p_id_2_topic_id and topic_id_2_fc_ids

    Returns
    -------
        neg_ids: list
            A list of n fact check ids to be used as negatives for the post
    """

    neg_ids = []

    # Create a log folder if it does not exist yet
    if not os.path.exists(LOG_FOLDERNAME):
        os.makedirs(LOG_FOLDERNAME)

    # Open the log file and write the header
    if not os.path.exists(os.path.join(LOG_FOLDERNAME, "negatives-topic-5.log")):
        log_file = open(os.path.join(LOG_FOLDERNAME, "negatives-topic-5.log"), "w")
        LOG_HEADER = "LOG FOR NEGATIVE SAMPLING BY TOPIC\n"
        topic_counts_str = ""
        sorted_topic_ids = sorted(list(topics_dict["topic_id_2_fc_ids"].keys()))
        for i in range(len(sorted_topic_ids)):
            topic_counts_str += str(sorted_topic_ids[i]) + ":" + str(len(topics_dict["topic_id_2_fc_ids"][sorted_topic_ids[i]])) + "; "
        TOPICS_STATS = "#TOPICS: " + str(len(list(topics_dict["topic_id_2_fc_ids"].keys()))) + "\n" + topic_counts_str + "\n\n"
        log_file.write(LOG_HEADER + TOPICS_STATS)
    else:
        log_file = open(os.path.join(LOG_FOLDERNAME, "negatives-topic-5.log"), "a")

    # Get the text of the post
    post_text = p_ids_to_texts[post_id]

    # Get the topic id of the text
    post_topic = topics_dict["p_id_2_topic_id"][post_id]

    # Get fcs ids, both in-topic and no-topic ones
    # The latter to be used as a fallaback in the case not enough fcs are available to sample
    FALLBACK_TOPIC_ID = -1
    if post_topic in topics_dict["topic_id_2_fc_ids"]:
        in_topic_fcs = topics_dict["topic_id_2_fc_ids"][post_topic]
    else: # there are topic ids (e.g., topic #68) that are not included
        post_topic = FALLBACK_TOPIC_ID
        in_topic_fcs = topics_dict["topic_id_2_fc_ids"][FALLBACK_TOPIC_ID]
    fallback_fcs = topics_dict["topic_id_2_fc_ids"][FALLBACK_TOPIC_ID]

    # From both lists, get only non-negative fcs, i.e., the actual potential ones
    potential_in_topic_fcs = list(set(in_topic_fcs).intersection(set(curr_cands)))
    potential_fallback_fcs = list(set(fallback_fcs).intersection(set(curr_cands)))

    # print(f"We need to sample {n} negative fcs for post id {post_id}.")
    # print(f"=> In-topic list has {len(potential_in_topic_fcs)} fcs (prev. {len(in_topic_fcs)}).")
    # print(f"=> Fallback list has {len(potential_fallback_fcs)} fcs (prev. {len(fallback_fcs)}).")

    log_file.write("p_id:" + str(post_id) + " t_id:" + str(post_topic) + " (size: " + str(len(potential_in_topic_fcs)) + ")\n" + post_text + "\n\n")

    # Sample n fact check ids from the negative candidates list without repetition
    if len(potential_in_topic_fcs) < n:
        log_file.write(str(len(potential_in_topic_fcs)) + " fact-checks from in-topic list:\n")
        neg_ids = random.sample(potential_in_topic_fcs, len(potential_in_topic_fcs))
        for neg_fc_id in neg_ids:
            log_file.write("=> " + "fc_id:" + str(neg_fc_id) + "\t" + fc_ids_to_texts[neg_fc_id] + "\n")
        
        log_file.write(str(n-len(potential_in_topic_fcs)) + " fact-checks from fallback list:\n")
        extra_neg_ids = random.sample(potential_fallback_fcs, n-len(potential_in_topic_fcs))
        for extra_neg_fc_id in extra_neg_ids:
            log_file.write("=> " + "fc_id:" + str(extra_neg_fc_id) + "\t" + fc_ids_to_texts[extra_neg_fc_id] + "\n")
        neg_ids = neg_ids + extra_neg_ids
    else:
        log_file.write(str(n) + " fact-checks from in-topic list:\n")
        neg_ids = random.sample(potential_in_topic_fcs, n)
        for neg_fc_id in neg_ids:
            log_file.write("=> " + "fc_id:" + str(neg_fc_id) + "\t" + fc_ids_to_texts[neg_fc_id] + "\n")

    log_file.write("\n" + str(neg_ids))
    log_file.write("\n==========\n\n")
    log_file.close()

    return neg_ids


def get_negatives_by_similarity(
    post_id, curr_cands, n, p_ids_to_texts, fc_ids_to_texts, model, fc_ids_list, fc_embs_tensor):
    """
    A function that samples n (k*factor) negative fact check ids based on
    embeddings similarity (embeddings computed by using the TEMB_MODEL model) 
    from a list of candidate fact checks ids for a given post id.

    Parameters
    ----------
        post_id: int
            The id of the post
        curr_cands: list
            The fact check ids of the negative candidates
        n: int
            The number of negatives to sample (i.e., k*factor)
        p_ids_to_texts: dict
            A dictionary of post texts with id as key
        fc_ids_to_texts: dict
            A dictionary of fact check texts with id as key
        model: sentence_transformers model
            A sentence_transformers text embedding model
        fc_ids_list: list
            A list of fact check ids (matching indexes of fc_embs_tensor)
        fc_embs_tensor: dict
            A tensor of fact check embeddings (matching indexes of fc_ids)

    Returns
    -------
        neg_ids: list
            A list of n fact check ids to be used as negatives for the post
    """

    neg_ids = []

    # Create a log folder if it does not exist yet
    if not os.path.exists(LOG_FOLDERNAME):
        os.makedirs(LOG_FOLDERNAME)

    # Open the log file and write the header
    log_file = open(os.path.join(LOG_FOLDERNAME, "negatives-similarity-5.log"), "a")
    if not os.path.exists(os.path.join(LOG_FOLDERNAME, "negatives-similarity-5.log")):
        LOG_HEADER = "LOG FOR NEGATIVE SAMPLING BY SIMILARITY\n\n"
        log_file.write(LOG_HEADER)

    # Get the text of the post
    # post_text = ME5_PREFIX + p_ids_to_texts[post_id]
    post_text = p_ids_to_texts[post_id]
    log_file.write("p_id:" + str(post_id) + "\n" + post_text + "\n\n")

    # Compute text embeddings for the post
    post_emb = model.encode(post_text, convert_to_tensor=True)

    # Compute cosine similarity between the post and fact checks
    sim_scores = util.cos_sim(post_emb, fc_embs_tensor)[0]

    # Get the highest scoring (n*10) fact checks
    # 10 is used to ensure all non-positive fact checks can be actually found
    scores, indices = torch.topk(sim_scores, k=n*10)

    # Iterate through the results, log them to a file, and store them to neg_ids
    # Note that n*10 iterations are envisioned (worst case scenario) but the
    # computation stops when n non-positive/gold fact-checks are actually found
    curr_fc_count = 0
    for score, idx in zip(scores, indices):
        # Stop the iteration if n non-positive fact checks have been found already
        if curr_fc_count >= n:
            break

        # Get details on the fact check and ensure it is not a positive one (not in curr_cands)
        curr_fc_id_neg = fc_ids_list[idx]
        curr_fc_text_neg = fc_ids_to_texts[fc_ids_list[idx]]
        curr_fc_score_neg = f"{score:.4f}"
        if curr_fc_id_neg in curr_cands:
            log_file.write("=> " + "fc_id:" + str(curr_fc_id_neg) + " (" + curr_fc_score_neg + ")\t" + curr_fc_text_neg + "\n")
            neg_ids.append(curr_fc_id_neg)
            curr_fc_count += 1
        else:
            log_file.write("[GOLD] => " + "fc_id:" + str(curr_fc_id_neg) + " (" + curr_fc_score_neg + ")\t" + curr_fc_text_neg + "\n")
    log_file.write("\n" + str(neg_ids))
    log_file.write("\n==========\n\n")
    log_file.close()

    # Ensure we had n fact checks to be used as negatives for the post
    if curr_fc_count < n:
        sys.exit("ERROR. No enough fact checks have been found for post id {post_id}. Consider increasing the torch.topk() k parameter.")

    return neg_ids


def get_negatives_randomly(post_id, curr_cands, n, factor):
    """
    A function that samples n (k*factor) negative fact check ids at random
    from a list of candidate fact checks ids for a given post id.

    Parameters
    ----------
        post_id: int
            The id of the post
        curr_cands: list
            The fact check ids of the negative candidates
        n: int
            The number of negatives to sample (i.e., k*factor)
        k: int
            The number of negatives that determines the n value
        factor: int
            The factor that determines the n value

    Returns
    -------
        neg_ids: list
            A list of n fact check ids to be used as negatives for the post
    """

    # Sample n fact check ids from the negative candidates list without repetition
    # neg_ids = random.sample(curr_cands, n)

    # @Edit for being reproducible with previous experiments with k=5
    ################################################################################
    # Sample n fact check ids from the negative candidates list without repetition
    neg_ids = random.sample(curr_cands, 5*factor)

    return neg_ids


def get_pos_p_ids(train_pos):
    """
    A function that returns a list of unique post ids given a dictionary
    of training post-factcheck(s) pairs.

    Parameters
    ----------
        train_pos: dict
            A dictionary of (positive) training post-factcheck(s) pairs

    Returns
    -------
        list(p_ids_pos): list
            A list of unique post ids
    """

    p_ids_pos = set()

    for p_id, fc_ids in train_pos.items():
        p_ids_pos.add(p_id)
    
    return list(p_ids_pos)


def get_pos_fc_ids(train_pos):
    """
    A function that returns a list of unique fact check ids given a dictionary
    of training post-factcheck(s) pairs.

    Parameters
    ----------
        train_pos: dict
            A dictionary of (positive) training post-factcheck(s) pairs

    Returns
    -------
        list(fc_ids_pos): list
            A list of unique fact check ids
    """

    fc_ids_pos = set()

    for p_id, fc_ids in train_pos.items():
        for fc_id in fc_ids:
            fc_ids_pos.add(fc_id)
    
    # Sorting for reproducibility - sets do not rely on the seed
    return sorted(list(fc_ids_pos))


def get_train_positive_pairs(data_folder_path):
    """
    A function that returns a dictionary of positive training post-factcheck(s) 
    pairs in the form {post_id: [fc_id1, ..., fc_idN], ...}.

    Parameters
    ----------
        data_folder_path: str
            The path to the folder containing the .csv data files to process

    Returns
    -------
        train_pos: dict
            A dictionary of (positive) training post-factcheck(s) pairs
    """

    train_pos = {} # dictionary storing positive pairs 
    # Note that all relationships in data are "backlink" / "claimreview_schema"

    with open(os.path.join(data_folder_path, MAPPINGS_FILENAME), "r", newline='') as f:
        data_reader = csv.reader(f, delimiter=',', quotechar='"')
        next(data_reader) # skip the header line

        for row in data_reader:
            post_id = int(row[0])
            fc_id = int(row[1])
            relationship = row[2]
            split = row[3]

            # Case of interest: the pair is not similarity:identical and belongs to the training set
            if relationship != "similarity:identical":
                if split == "train":
                    if post_id not in train_pos.keys():
                        train_pos[post_id] = [fc_id]
                    else:
                        train_pos[post_id].append(fc_id)
                # Other cases: the pair is either part of the validation or test set (skip)
                elif split == "dev":
                    continue
                elif split == "test":
                    continue
                # Corner case: there is an error in the split column
                else:
                    sys.exit(f"ERROR. The \"{split}\" split is not foreseen.")

    return train_pos


def get_training_posts(data_folder_path, p_ids_pos):
    """
    A function that creates a dictionary of id and text for all
    posts for further processing.

    Parameters
    ----------
        data_folder_path: str
            The path to the folder containing the .csv data files to process
        p_ids_pos: list
            A list of unique post ids

    Returns
    -------
        p_ids_to_texts: dict
            A dictionary of post texts with id as key
    """

    p_ids_to_texts = {}

    with open(os.path.join(data_folder_path, POSTS_FILENAME), "r", newline='') as f:
        data_reader = csv.reader(f, delimiter=',', quotechar='"')
        next(data_reader) # skip the header line

        for row in tqdm(data_reader):
            p_id = int(row[0])

            # Consider only posts that are linked to some fact checks in the training set
            if p_id in p_ids_pos:
                p_text = row[2]
                if p_id not in p_ids_to_texts.keys():
                    p_ids_to_texts[p_id] = p_text
                else:
                    sys.exit(f"ERROR. Post id {p_id} is in the dictionary already.")

    return p_ids_to_texts


def get_training_fact_checks(data_folder_path, fc_ids_pos):
    """
    A function that creates a dictionary of id and text for all
    (positive) training fact checks for further processing.

    Parameters
    ----------
        data_folder_path: str
            The path to the folder containing the .csv data files to process
        fc_ids_pos: list
            A list of unique fact check ids

    Returns
    -------
        fc_ids_to_texts: dict
            A dictionary of fact check texts with id as key
    """

    fc_ids_to_texts = {}

    with open(os.path.join(data_folder_path, FACTCHECKS_FILENAME), "r", newline='') as f:
        data_reader = csv.reader(f, delimiter=',', quotechar='"')
        next(data_reader) # skip the header line

        for row in tqdm(data_reader):
            fc_id = int(row[0])

            # Consider only fact checks that are linked to some post in the training set
            if fc_id in fc_ids_pos:
                fc_text = row[3] # we use the claim field
                if fc_id not in fc_ids_to_texts.keys():
                    fc_ids_to_texts[fc_id] = fc_text
                else:
                    sys.exit(f"ERROR. Fact check id {fc_id} is in the dictionary already.")

    return fc_ids_to_texts


def create_fact_checks_embeddings(data_folder_path, fc_ids_pos, fc_ids_to_texts, model):
    """
    A function that creates embeddings for (positive) training 
    fact checks and serialize the dict object containing them.

    Parameters
    ----------
        data_folder_path: str
            The path to the folder containing the .csv data files to process
        fc_ids_pos: list
            A list of unique fact check ids
        fc_ids_to_texts: dict
            A dictionary of fact check texts with id as key
        model: sentence_transformers model
            A sentence_transformers text embedding model
    """

    fc_embs = {} # a dictionary of fact check embeddings with id as key

    # For each fact check, compute its text embeddings and store the results
    print(f"Computing embeddings for all fact checks...")
    for fc_id, fc_text in tqdm(fc_ids_to_texts.items()):
        # fc_emb = model.encode(ME5_PREFIX + fc_text, convert_to_tensor=True)
        fc_emb = model.encode(fc_text, convert_to_tensor=True)
        fc_embs[fc_id] = fc_emb
    print("=> Done.")

    # Serialize fact check embeddings to a file
    with open(os.path.join(RES_FOLDERNAME, EMBEDDINGS_FILENAME), "wb") as handle:
        pickle.dump(fc_embs, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_topics_file(p_ids_to_texts, fc_ids_to_texts):
    """
    A function that assigns topic ids for both posts and (positive)
    training fact checks and serialize the dict object containing them.

    Parameters
    ----------
        p_ids_to_texts: dict
            A dictionary of post texts with id as key
        fc_ids_to_texts: dict
            A dictionary of fact check texts with id as key
    """

    p_id_2_topic_id = {}    # dictionary storing post id to topic id mappings
    topic_id_2_fc_ids = {}  # dictionary storing topic id to fact check ids mappings

    # Separate post and fact check ids from text for computation and later retrieval
    p_ids = list(p_ids_to_texts.keys())#[:100]
    p_texts = list(p_ids_to_texts.values())#[:100]
    fc_ids = list(fc_ids_to_texts.keys())#[:100]
    fc_texts = list(fc_ids_to_texts.values())#[:100]

    # Use both post and fact check texts for topic modeling (i.e., concatenate them)
    all_texts = p_texts + fc_texts

    print(f"Computing topics for all posts and fact checks...")

    # 1) Extract embeddings [CUSTOM TEMB_MODEL]
    embedding_model = SentenceTransformer(TEMB_MODEL)

    # 2) Reduce dimensionality [Default hyperparameters]
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

    # 3) Cluster reduced embeddings [Default hyperparameters]
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    # 4) Tokenize topics [STOPWORDS-ISO MULTILINGUAL stop_words INSTEAD OF ENGLISH ONES]
    #    Get a list of multilingual stopwords (note: a small number of languages has no stopwords)
    multilingual_sws = list(stopwords.stopwords(
        ["af", "ar", "bg", "bn", "ca", "co", "cs", "da", "de", "el", "en", "es", "et", "fa", "fi", "fr", "hi", "hr", "hu", "id", "it", "ko", "lv", "mr", "ms", "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl", "so", "sv", "th", "tl", "tr", "uk", "ur", "zh"]))
    vectorizer_model = CountVectorizer(stop_words=multilingual_sws) # @IMPR: tokenizer

    # 5) Create topic representations [Default hyperparameters]
    ctfidf_model = ClassTfidfTransformer()

    # All steps together [Default]
    topic_model = BERTopic(
        embedding_model=embedding_model,    # Extract embeddings
        umap_model=umap_model,              # Reduce dimensionality
        hdbscan_model=hdbscan_model,        # Cluster reduced embeddings
        vectorizer_model=vectorizer_model,  # Tokenize topics
        ctfidf_model=ctfidf_model           # Extract topic words
    )
    # DEFAULTS:
    # BERTopic(
    #     calculate_probabilities=False, 
    #     ctfidf_model=ClassTfidfTransformer(...), 
    #     embedding_model=None, 
    #     hdbscan_model=HDBSCAN(...), 
    #     language=english, 
    #     low_memory=False, 
    #     min_topic_size=10, 
    #     n_gram_range=(1, 1), 
    #     nr_topics=None, 
    #     representation_model=None, 
    #     seed_topic_list=None, 
    #     top_n_words=10, 
    #     umap_model=UMAP(...), 
    #     vectorizer_model=CountVectorizer(...), 
    #     verbose=False, 
    #     zeroshot_min_similarity=0.7, 
    #     zeroshot_topic_list=None
    # )

    # Feed the data to the topic model and get a list of topic ids associated with texts
    topics, probs = topic_model.fit_transform(all_texts)
    # print(topic_model.get_topic_info())

    assert len(topics) == (len(p_ids) + len(fc_ids)) # (len(p_ids[:100]) + len(fc_ids[:100]))

    # Get dictionary for post id to topics id
    for i in range(len(p_ids)): # (len(p_ids[:100])):
        # print(str(p_ids[i]) + "\t" + str(topics[i]) + "\t" + p_texts[i])
        p_id_2_topic_id[p_ids[i]] = topics[i]

    # Get dictionary for topics id to fact check ids
    for i in range(len(fc_ids)): # len(fc_ids[:100])):
        # print(str(fc_ids[i]) + "\t" + str(topics[len(p_ids[:100])+i]) + "\t" + fc_texts[i])
        if topics[len(p_ids)+i] not in topic_id_2_fc_ids.keys(): # topics[len(p_ids[:100])+i]
            topic_id_2_fc_ids[topics[len(p_ids)+i]] = [fc_ids[i]] # topics[len(p_ids[:100])+i]
        else:
            topic_id_2_fc_ids[topics[len(p_ids)+i]].append(fc_ids[i]) # topics[len(p_ids[:100])+i]

    print("=> Done.")

    # Merge the dicts into a single one and serialize it
    topics = {}
    topics["p_id_2_topic_id"] = p_id_2_topic_id
    topics["topic_id_2_fc_ids"] = topic_id_2_fc_ids

    # Serialize topics dictionary to a file
    with open(os.path.join(RES_FOLDERNAME, TOPICS_FILENAME), "wb") as handle:
        pickle.dump(topics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # THis WARNING appeared during topic modeling but it does not affect the results:
    # huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    # To disable this warning, you can either:
    # - Avoid using `tokenizers` before the fork if possible
    # - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


def get_train_negative_pairs(data_folder_path, train_pos, p_ids_pos, fc_ids_pos, strategy, k):
    """
    A function that returns a dictionary of negative training post-factcheck(s) 
    pairs in the form {post_id: [fc_id1, ..., fc_idN], ...}, sampled based on
    a given strategy and a negative-to-positive ratio k. Positive pairs and
    fact check ids are used in this function to ensure the process avoids
    sampling (as negatives) actual positive pairs.

    Parameters
    ----------
        train_pos: dict
            A dictionary of (positive) training post-factcheck(s) pairs
        p_ids_pos: list
            A list of unique post ids
        fc_ids_pos: list
            A list of unique fact check ids
        strategy: str
            The strategy to apply for sampling negative pairs
        k: int
            The ratio of negatives to be sampled (compared to the positives)

    Returns
    -------
        train_neg: dict
            A dictionary of (negative) training post-factcheck(s) pairs
    """

    train_neg = {} # dictionary storing negative pairs
    # factors = [] # debug: for getting the 1:n distrubution

    print(f"Sampling negative examples using \"{strategy}\" strategy (k={k})...")

    if strategy == "similarity":
        # Load the text embedding model
        print(f"Loading the {TEMB_MODEL} text embedding model...")
        model = SentenceTransformer(TEMB_MODEL)
        print("=> Done.")

        # Get a dictionary of posts in the form {id: text, ...}
        print(f"Loading the post texts...")
        p_ids_to_texts = get_training_posts(data_folder_path, p_ids_pos)
        print("=> Done.")

        # Get a dictionary of fact checks in the form {id: text, ...}
        print(f"Loading the fact check texts...")
        fc_ids_to_texts = get_training_fact_checks(data_folder_path, fc_ids_pos)
        print("=> Done.")

        # Create a res folder if it does not exist yet
        if not os.path.exists(RES_FOLDERNAME):
            os.makedirs(RES_FOLDERNAME)

        # If the embeddings file has not been created yet, create it
        if not os.path.exists(os.path.join(RES_FOLDERNAME, EMBEDDINGS_FILENAME)):
            create_fact_checks_embeddings(data_folder_path, fc_ids_pos, fc_ids_to_texts, model)

        # Read the embeddings file
        print("Loading fact check embeddings...")
        with open(os.path.join(RES_FOLDERNAME, EMBEDDINGS_FILENAME), 'rb') as handle:
            fc_embs = pickle.load(handle)
        print("=> Done.")

        # Separate ids and embeddings for easing computation and retrieval at a later stage
        fc_ids_list = list(fc_embs.keys())
        fc_embs_tensor = torch.stack(list(fc_embs.values()), dim=0)

    if strategy == "topic":
        # Get a dictionary of posts in the form {id: text, ...}
        print(f"Loading the post texts...")
        p_ids_to_texts = get_training_posts(data_folder_path, p_ids_pos)
        print("=> Done.")

        # Get a dictionary of fact checks in the form {id: text, ...}
        print(f"Loading the fact check texts...")
        fc_ids_to_texts = get_training_fact_checks(data_folder_path, fc_ids_pos)
        print("=> Done.")

        # Create a res folder if it does not exist yet
        if not os.path.exists(RES_FOLDERNAME):
            os.makedirs(RES_FOLDERNAME)

        # If the topics file has not been created yet, create it
        if not os.path.exists(os.path.join(RES_FOLDERNAME, TOPICS_FILENAME)):
            create_topics_file(p_ids_to_texts, fc_ids_to_texts)

        # Read the topics file
        print("Loading topics dictionary...")
        with open(os.path.join(RES_FOLDERNAME, TOPICS_FILENAME), 'rb') as handle:
            topics_dict = pickle.load(handle)
        print("=> Done.")

    # Iterate over each post to sample negatives for it
    for p_id, fc_ids in tqdm(train_pos.items()):
        # Get multiplication factor by counting positive fcs for the current post
        factor = len(fc_ids)
        # factors.append(factor) # debug: for getting the 1:n distrubution

        # Get negative candidates (i.e., diff btw all fcs and positive ones)
        curr_cands = [fc_id for fc_id in fc_ids_pos if fc_id not in fc_ids]

        # Sample k*factor negatives from the candidates based on the strategy
        if strategy == "random":
            neg_ids = get_negatives_randomly(p_id, curr_cands, k*factor, factor)
            train_neg[p_id] = [neg for neg in neg_ids]
        elif strategy == "similarity":
            neg_ids = get_negatives_by_similarity(
                p_id, curr_cands, k*factor, p_ids_to_texts, fc_ids_to_texts, model, fc_ids_list, fc_embs_tensor)
            train_neg[p_id] = [neg for neg in neg_ids]
        elif strategy == "topic":
            neg_ids = get_negatives_by_topic(
                p_id, curr_cands, k*factor, p_ids_to_texts, fc_ids_to_texts, topics_dict)
            train_neg[p_id] = [neg for neg in neg_ids]
        else:
            sys.exit(f"ERROR. Strategy \"{strategy}\" is not implemented yet.")

    # @Edit for being reproducible with previous experiments with k=5
    ################################################################################
    # Sample the remaining n fact check ids
    if strategy == "random":
        for p_id, fc_ids in tqdm(train_pos.items()):
            factor = len(fc_ids)
            curr_cands = [fc_id for fc_id in fc_ids_pos if fc_id not in fc_ids]
            remaining_cands = [x for x in curr_cands if x not in train_neg[p_id]]
            neg_ids_new = random.sample(remaining_cands, 15*factor)
            train_neg[p_id] = train_neg[p_id] + neg_ids_new

    print("=> Done.")

    # p_to_fc_counter = Counter(factors) # debug: for getting the 1:n distrubution
    # Counter({1: 39750, 2: 4592, 3: 833, 4: 218, 5: 55, 6: 12, 7: 6, 8: 2, 9: 2, 10: 2, 18: 1})
    # OLD:
    # 1:1 => 44872 (76%; cumulative: 76%)
    # 1:2 => 11379 (19%; cumulative: 95%)
    # 1:3 => 2101 (4%; cumulative: 99%)
    # 1:4 => 600
    # 1:5 => 173
    # 1:6 => 42
    # 1:7 => 20
    # 1:8 => 13
    # 1:9 => 7
    # 1:10 => 4
    # 1:11 => 4
    # 1:12 => 1
    # 1:13 => 1
    # 1:20 => 1
    # 1:21 => 1
    # 1:41 => 1

    return train_neg


def create_negative_examples(data_folder_path, strategy, neg_to_pos_ratio):
    """
    A function that coordinates the sampling of negative training examples
    (i.e., pairs) based on the available positive training examples and given
    a strategy and a negative-to-positive ratio. It then writes the results
    in an output file in a format similar to "fact_check_post_mapping.csv".

    Parameters
    ----------
        data_folder_path: str
            The path to the folder containing the .csv data files to process
        strategy: str
            The strategy to apply for sampling negative pairs
        neg_to_pos_ratio: int
            The ratio of negatives to be sampled (compared to the positives)
    """

    # Get a dictionary of (positive) training post-factcheck(s) pairs {post_id: [fc_id1, ..., fc_idN], ...}
    train_pos = get_train_positive_pairs(data_folder_path)

    # BEFORE: Number of (positive) post-factcheck associations: |P-FC|: before: 77948]
    # pairs_counter = 0
    # for fc_id_list in train_pos.values():
    #     assert len(fc_id_list) == len(set(fc_id_list)) # ensure no fc_id is repeated (duplicates)
    #     for fc_id in set(fc_id_list): pairs_counter += 1
    # print(len(train_pos.keys())) # 45473 (47850) TRAIN unique keys (post-(all factchecks) associations) - in par considering identical ones
    # print(pairs_counter)         # 56394 (52766) TRAIN fact-checks - in par considering identical ones

    # Get a list of the unique post ids
    p_ids_pos = get_pos_p_ids(train_pos)    # |P| = 45473 (before it was 59220)

    # Get a list of the unique fact-check ids
    fc_ids_pos = get_pos_fc_ids(train_pos)  # [F| = 41964 (before it was 55750)

    # Get a dictionary of (negative) training post-factcheck(s) pairs {post_id: [fc_id1, ..., fc_idN], ...}
    train_neg = get_train_negative_pairs(
        data_folder_path, train_pos, p_ids_pos, fc_ids_pos, strategy, neg_to_pos_ratio)

    # Create the output .csv file for the specific strategy and neg_to_pos_ratio
    create_output_file(data_folder_path, train_neg, strategy, neg_to_pos_ratio)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--data_folder_path", type=str, default="dataset_with_splits", 
        help="The path to the folder containing the MultiClaim dataset files (version 2024-03).")
    parser.add_argument("-S", "--strategy", type=str, default="random", 
        choices=["random", "similarity", "topic"], help="The strategy to be used \
        for sampling negatives. Choices: [random, similarity, topic]. Default: random.")
    parser.add_argument("-R", "--neg_to_pos_ratio", type=int, default=1, 
        choices=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        help="The ratio of negative examples to be sampled for each positive one \
        (e.g., 1: 1 neg for each pos (1:1); 2: 2 neg for each pos (1:2)). Choices: \
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]. Default: 1.")
    args = parser.parse_args()

    # Create negative examples and the resulting output file
    create_negative_examples(args.data_folder_path, args.strategy, args.neg_to_pos_ratio)
