import { createAsyncThunk } from "@reduxjs/toolkit";
import { authorizedCustomAxios } from "customAxios";
import { homeActions } from "app/store/ducks/home/homeSlice";
import {
    ExtraArticleProps,
    RecentArticlesProps,
} from "app/store/ducks/home/homeThunk.type";
import { FAIL_TO_REISSUE_MESSAGE } from "utils/constant";
import { authAction } from "app/store/ducks/auth/authSlice";
import Papa from 'papaparse';

export const getHomeArticles = createAsyncThunk<PostType.ArticleStateProps[]>(
    "home/getHomeArticles",
    async (payload, ThunkOptions) => {
        try {
            const {
                data: { data },
            }: RecentArticlesProps = await authorizedCustomAxios.get(
                `/posts/recent`,
            );
            const articlesState: PostType.ArticleStateProps[] = data.map(
                (article) => ({
                    ...article,
                    followLoading: false,
                }),
            );
            return articlesState;
        } catch (error) {
            // error === FAIL_TO_REISSUE_MESSAGE &&
            //     ThunkOptions.dispatch(authAction.logout());
            throw ThunkOptions.rejectWithValue(error);
        }
    },
);

interface Profile {
    id: number,
    gender: string,
    name: string,
    username: string,
    image_name: string,
    image_uuid: string
}

const parseCSV = (csvText : string): any => {
    return new Promise((resolve, reject) => {
        Papa.parse(csvText, {
            header: true, // Adjust this based on your CSV structure
            skipEmptyLines: true,
            complete: (results) => resolve(results.data),
            error: (error : any) => reject(error),
        });
    });
};

const getArticle = (currentuser: string, profile: Profile): PostType.ArticleStateProps => {
    const followingMemberUsernameLikedPost = null; // 내가 팔로우한 사람 중에서 이 글을 좋아한 사람 있으면 보내줌
    const member = {
        id: profile.id,
        username: profile.username,
        name: profile.name,
        gender: profile.gender,
        image: {
            imageUrl: `profiles/${profile.image_name}`,
            imageType: 'jpg',
            imageName: `profile-${profile.username}`,
            imageUUID: profile.image_uuid
        },
        hasStory: false,
    };
    const postBookmarkFlag = false; // 내가 북마크 했는지
    const postCommentsCount = 0;
    const postContent =  'The photo of mine';
    const postId =  profile.id;
    const postImages = [
        {
            id: profile.id,
            postImageUrl: `profiles/${profile.image_name}`,
            postTags: [],
            altText: profile.image_name,
        }
    ];
    const postLikeFlag = false; // 내가 좋아요 했는지
    const postLikesCount = 0;
    const postUploadDate = '2025-01-10';
    const hashtagsOfContent : any = [];
    const mentionsOfContent : any = [];
    const likeOptionFlag = false; // 업로드한 사람만 좋아요 및 좋아요한 사람 확인 가능
    const commentOptionFlag = true; // 댓글 작성 가능 여부
    const following = false;
    const recentComments : any = [];
    const followLoading = false;
    const filtered = true;

    return {
        followingMemberUsernameLikedPost: followingMemberUsernameLikedPost,
        member: member,
        postBookmarkFlag: postBookmarkFlag, // 내가 북마크 했는지
        postCommentsCount: postCommentsCount,
        postContent: postContent,
        postId: postId,
        postImages: postImages,
        postLikeFlag: postLikeFlag, // 내가 좋아요 했는지
        postLikesCount: postLikesCount,
        postUploadDate: postUploadDate,
        hashtagsOfContent: hashtagsOfContent,
        mentionsOfContent: mentionsOfContent,
        likeOptionFlag: likeOptionFlag, // 업로드한 사람만 좋아요 및 좋아요한 사람 확인 가능
        commentOptionFlag: commentOptionFlag, // 댓글 작성 가능 여부
        following: following,
        recentComments: recentComments,
        followLoading: followLoading,
        filtered: filtered,
    }
}

export const getArticles = createAsyncThunk<PostType.ArticleStateProps[]>(
    "home/getHomeArticles",
    async (payload, ThunkOptions) => {
        try {
            const response = await fetch('/profile.csv');
            const res_text = await response.text();
            const profiles = await parseCSV(res_text) ;
            const new_articles: PostType.ArticleStateProps[] = [];

            for (const profile of profiles) {
                const article = getArticle('alexander.adams', profile)
                new_articles.push(article)
            }

            const articlesState: PostType.ArticleStateProps[] = new_articles
            return articlesState;
        } catch (error) {
            // error === FAIL_TO_REISSUE_MESSAGE &&
            //     ThunkOptions.dispatch(authAction.logout());
            throw ThunkOptions.rejectWithValue(error);
        }
    },
);

export const getExtraArticle = createAsyncThunk<
    PostType.ArticleStateProps,
    {
        page: number;
    }
>("home/getExtraArticle", async (payload, ThunkOptions) => {
    const config = {
        params: {
            page: payload.page,
        },
    };
    try {
        const {
            data: {
                data: { content: data, empty },
            },
        }: ExtraArticleProps = await authorizedCustomAxios.get(
            `/posts`,
            config,
        ); // 단건 조회 api 추가

        if (empty) {
            throw ThunkOptions.rejectWithValue(
                "게시물이 더 이상 존재하지 않습니다.",
            );
        }
        ThunkOptions.dispatch(homeActions.increaseExtraArticlesCount());
        const articleState: PostType.ArticleStateProps = {
            ...data[0],
            // isFollowing: true,
            followLoading: false,
        };
        return articleState;
    } catch (error) {
        error === FAIL_TO_REISSUE_MESSAGE &&
            ThunkOptions.dispatch(authAction.logout());
        throw ThunkOptions.rejectWithValue(error);
    }
});

export const postUnfollow = createAsyncThunk<
    string, // 이후 데이터 보고 수정
    {
        username: string;
    }
>("home/postUnfollow", async (payload, ThunkOptions) => {
    try {
        const {
            data: { data },
        } = await authorizedCustomAxios.delete(`/${payload.username}/follow`);

        return data;
    } catch (error) {
        error === FAIL_TO_REISSUE_MESSAGE &&
            ThunkOptions.dispatch(authAction.logout());
        throw ThunkOptions.rejectWithValue(error);
    }
});

export const postFollow = createAsyncThunk<
    string, // 이후 데이터 보고 수정
    {
        username: string;
    }
>("home/postFollow", async (payload, ThunkOptions) => {
    try {
        const {
            data: { data: isSuccess },
            data,
        } = await authorizedCustomAxios.post(
            `/${payload.username}/follow`,
            null,
        );
        if (
            data.code === "F006" ||
            data.message === "팔로우할 수 없는 대상입니다."
        ) {
            throw ThunkOptions.rejectWithValue("차단");
        }
        return isSuccess;
    } catch (error) {
        error === FAIL_TO_REISSUE_MESSAGE &&
            ThunkOptions.dispatch(authAction.logout());
        throw ThunkOptions.rejectWithValue(error);
    }
});

export const postLike = createAsyncThunk<
    {
        status: number;
        code: string;
        message: string;
        errors?: [];
        data?: {
            status: boolean;
        };
    },
    // any,
    { postId: number }
>("home/postLike", async (payload, ThunkOptions) => {
    const config = {
        params: {
            postId: payload.postId,
        },
    };
    try {
        const { data } = await authorizedCustomAxios.post(
            `/posts/like`,
            null,
            config,
        );
        return data;
    } catch (error) {
        error === FAIL_TO_REISSUE_MESSAGE &&
            ThunkOptions.dispatch(authAction.logout());
        throw ThunkOptions.rejectWithValue(error);
    }
});

export const deleteLike = createAsyncThunk<
    {
        status: number;
        code: string;
        message: string;
        errors?: [];
        data?: {
            status: boolean;
        };
    },
    { postId: number }
>("home/deleteLike", async (payload, ThunkOptions) => {
    const config = {
        params: {
            postId: payload.postId,
        },
    };
    try {
        const { data } = await authorizedCustomAxios.delete(
            `/posts/like`,
            config,
        );
        return data;
    } catch (error) {
        error === FAIL_TO_REISSUE_MESSAGE &&
            ThunkOptions.dispatch(authAction.logout());
        throw ThunkOptions.rejectWithValue(error);
    }
});
