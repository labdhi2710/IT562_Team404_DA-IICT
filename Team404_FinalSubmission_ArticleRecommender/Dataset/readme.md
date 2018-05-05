Description on the data files:

-shared_articles.csv: 

Contains information about the articles shared in the platform. Each article has its sharing date (timestamp), the original url, title, content in plain text, the article' lang (Portuguese: pt or English: en) and information about the user who shared the article (author).

There are two possible event types at a given timestamp:

    CONTENT SHARED: The article was shared in the platform and is available for users.
    CONTENT REMOVED: The article was removed from the platform and not available for further recommendation.

For the sake of simplicity, we only consider here the "CONTENT SHARED" event type, assuming (naively) that all articles were available during the whole one year period. For a more precise evaluation (and higher accuracy), only articles that were available at a given time should be recommended.

-users_interaction.csv

Contains logs of user interactions on shared articles. It can be joined to articles_shared.csv by contentId column.

The eventType values are:

    VIEW: The user has opened the article.
    LIKE: The user has liked the article.
    COMMENT CREATED: The user created a comment in the article.
    FOLLOW: The user chose to be notified on any new comment in the article.
    BOOKMARK: The user has bookmarked the article for easy return in the future.
