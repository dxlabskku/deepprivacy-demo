import ImgHashTagAvatar from "components/Common/Article/ArticleImgSlider/ImgHashTagAvatar";
import ImgHashTagUsername from "components/Common/Article/ArticleImgSlider/ImgHashTagUsername";
import React, { useState, useEffect } from "react";
import styled from "styled-components";

const StyledArticleImgSliderUnit = styled.div`
    width: 100%;
    scroll-snap-align: center;
    position: relative;
    & > .avatar {
        position: absolute;
        bottom: 0;
        left: 0;
    }
    & > a {
    }
    & > img {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    &.containImgHashTags {
        cursor: pointer;
    }
`;

interface ArticleImgSliderUnitProps {
    imageDTO: CommonType.PostImageDTOProps;
    unitWidth: number;
    filtered: boolean;
    gender: string;
    following: boolean;
}

const ArticleImgSliderUnit = ({
    imageDTO,
    unitWidth,
    filtered,
    gender,
    following,
}: ArticleImgSliderUnitProps) => {
    const [isAvatarOn, setIsAvatarOn] = useState(false);
    const [isImgHashTagsOn, setIsImgHashTagOn] = useState<boolean | null>(null);
    const [timeoutId, setTimeoutId] = useState<null | ReturnType<
        typeof setTimeout
    >>(null);
    const [isFiltered, setIsFiltered] = useState(filtered);
    const [imageurl, setImageurl] = useState(imageDTO.postImageUrl);

    const onClickHandler = () => {
        if (!isAvatarOn) {
            setIsAvatarOn(true);
            setIsImgHashTagOn(true);
        } else if (!isImgHashTagsOn) {
            setIsImgHashTagOn(true);
        } else if (isImgHashTagsOn) {
            setIsImgHashTagOn(false);
        }
        setTimeoutId(null);
    };

    const differentiateClickEvents = () => {
        if (timeoutId === null) {
            setTimeoutId(setTimeout(onClickHandler, 500));
        } else {
            setTimeoutId(null);
            clearTimeout(timeoutId);
        }
    };

    const imgDoubleClickHandler = (): void => {
        if (!isAvatarOn) {
            setIsAvatarOn(true);
        }
    };

    useEffect(() => {
        if (isFiltered) {
            const unfilter = async () => {
                const image = await fetch(imageDTO.postImageUrl);
                const blob = await image.blob();
                const reader = new FileReader();
                reader.onloadend = async () => {
                    const result = reader.result;
                    const transformationOptions = ['blur', 'emoji', 'closest', 'furthest'];
                    const option = transformationOptions[Math.floor(transformationOptions.length * Math.random())];

                    const res = await fetch('http://115.145.36.214:8888/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({'image': result, 'gender': gender, 'method': option})
                    });
                    const res_json = await res.json();
                    const image = res_json[0];
                    setImageurl(`data:image/jpeg;base64,${image}`);
                    setIsFiltered(false);
                }
                reader.readAsDataURL(blob);
            };
            unfilter();
        }
        else{
            // console.log('isFiltered updated to false.')
        }
    }, [isFiltered]);

    return (
        <StyledArticleImgSliderUnit
            onClick={differentiateClickEvents}
            onDoubleClick={imgDoubleClickHandler}
            className={imageDTO.postTags ? "containImgHashTags" : ""}
        >
            <div className="avatar">
                {imageDTO.postTags && isAvatarOn && <ImgHashTagAvatar />}
            </div>
            {imageDTO.postTags &&
                isAvatarOn &&
                imageDTO.postTags.map((postTag) => (
                    <ImgHashTagUsername
                        key={postTag.id}
                        postTagDTO={postTag}
                        isImgHashTagsOn={isImgHashTagsOn}
                    />
                ))}
            <img
                src={isFiltered ? 'loading.gif' : following ? imageDTO.postImageUrl : imageurl}
                alt={isFiltered ? 'loading.gif' : following ? imageDTO.postImageUrl : imageurl}
                width={unitWidth}
            />
        </StyledArticleImgSliderUnit>
    );
};

export default ArticleImgSliderUnit;
