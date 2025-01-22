import { getHomeArticles, getArticles } from "app/store/ducks/home/homThunk";
import { useAppDispatch, useAppSelector } from "app/store/Hooks";
import Article from "components/Common/Article";
import ExtraLoadingCircle from "components/Home/ExtraLoadingCircle";
import { useEffect } from "react";

const user = 'lucas.thomas'

const HomeSection = () => {
    const { articles, isLoading, isExtraArticleLoading } = useAppSelector(
        (state) => state.home,
    );
    const dispatch = useAppDispatch();

    useEffect(() => {
        const fetchData = async () => {
            await dispatch(getArticles())
        };
        fetchData();
    }, [dispatch]);

    return (
        <section>
            {isLoading ||
                articles.slice(0, 1).map((article, index) => (
                    <Article
                        key={article.postId}
                        article={article}
                        isObserving={articles.length - 4 === index}
                    />
                ))}
            {isExtraArticleLoading && <ExtraLoadingCircle />}
        </section>
    );
};

export default HomeSection;
