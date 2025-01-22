import { uploadActions } from "app/store/ducks/upload/uploadSlice";
import { useAppDispatch, useAppSelector } from "app/store/Hooks";
import { ReactComponent as LeftArrow } from "assets/Svgs/leftArrow.svg";
import { ReactComponent as RightArrow } from "assets/Svgs/rightArrow.svg";
import EditCanvasUnit from "components/Common/Header/Upload/Edit/EditCanvasUnit";
import UploadHeader from "components/Common/Header/Upload/UploadHeader";
import React, { ChangeEvent, useMemo, useState } from "react";
import styled from "styled-components";
import { useEffect } from 'react';

const StyledEdit = styled.div`
    display: flex;
    width: 100%;
    min-width: ${348 + 340 + 400}px;
    height: 100%;
    min-height: ${348}px;
    & > .upload__imgCanvasLayout {
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: ${(props) => props.theme.color.bg_gray};
        position: relative;
        & > button {
            position: absolute;
            width: 32px;
            height: 32px;
            background-color: rgba(26, 26, 26, 0.8);
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 4px 12px rgb(0 0 0 / 15%);
            &:hover {
                background-color: rgba(26, 26, 26, 0.5);
            }
        }
        & > .left {
            left: 8px;
        }
        & > .right {
            right: 8px;
        }
    }
    & > .upload__imgEditor {
        width: 740px;
        min-width: 740px;
        max-width: 100%;
        border-left: 1px solid ${(props) => props.theme.color.bd_gray};
        & > .header {
            display: flex;
            height: 53px;
            width: 100%;
            & > div {
                flex: 1;
                text-align: center;
                font-size: 16px;
                line-height: 24px;
                padding: 14px 0;
                font-weight: ${(props) => props.theme.font.bold};
                border-bottom: 1px solid ${(props) => props.theme.color.bd_gray};
                color: ${(props) => props.theme.color.bd_gray};
                cursor: pointer;
                transition: all 0.2s;
                &.active {
                    border-bottom: 1px solid black;
                    color: black;
                }
            }
        }
        & > .deepprivacy {
            height: calc(100% - 53px);
            max-height: calc(100% - 53px);
            min-height: calc(100% - 53px);
            overflow-y: auto;

            & > .subtitle {
                margin-left: 24px;
                margin-top: 24px;
                font-size: 16px;
                line-height: 24px;
                font-weight: ${(props) => props.theme.font.bold};
                color: '#000000';
            }

            & > .image__row {
                width: calc(100% - 48px);
                margin-top: 12px;
                margin-left: 24px;
                display: flex;
                justify-content: space-between;
                & > .image__element {
                    width: 30%;
                }
                & > .image__element:hover {
                    border: 5px solid red;
                }
            }
        }
        & > .adjust__input {
            padding: 0 16px;
            & > div {
                width: 100%;
            }
            & > div:first-child {
                padding: 14px 0;
                display: flex;
                justify-content: space-between;
                line-height: 24px;
                & > button {
                    color: ${(props) => props.theme.color.blue};
                    display: none;
                    &.entered {
                        display: block;
                    }
                }
            }
            & > div:last-child {
                height: 36px;
                display: flex;
                align-items: center;
                & > input {
                    flex: 1;
                    // range background 제거
                    -webkit-appearance: none;
                    background: transparent;
                    border: none;
                    height: 2px;
                    &::-webkit-slider-thumb {
                        -webkit-appearance: none; // 동그라미 제거
                        background-color: black;
                        width: 20px;
                        height: 20px;
                        border-radius: 50%;
                    }
                    /* &::-moz-range-thumb {
                    } */
                }
                & > div {
                    width: 24px;
                    margin-left: 8px;
                    text-align: right;
                }
            }
        }
    }
`;

interface EditProps {
    currentWidth: number;
}

const MIN_WIDTH = 348;

const getCanvasSize = (
    ratioMode: UploadType.RatioType,
    processedCanvasLayoutWidth: number,
) => {
    switch (ratioMode) {
        case "square":
            return {
                width: processedCanvasLayoutWidth,
                height: processedCanvasLayoutWidth,
            };
        case "original":
            return {
                width: processedCanvasLayoutWidth,
                height: processedCanvasLayoutWidth / 1.93,
            };
        case "thin":
            return {
                width: processedCanvasLayoutWidth * 0.8,
                height: processedCanvasLayoutWidth,
            };
        case "fat":
            return {
                width: processedCanvasLayoutWidth,
                height: (processedCanvasLayoutWidth * 9) / 16,
            };
    }
};

// 기존 함수를 제거하고, Canvas에서 이미지를 뽑아내었을 때 원본 이미지의 해상도를 유지하기 위해 Canvas width, height를 정밀하게 정하는 함수를 작성했습니다.
export const getNewImageSizeBasedOnOriginal = (
    ratioMode: UploadType.RatioType,
    currentFile: UploadType.FileProps,
) => {
    const { width, height, imageRatio } = currentFile;
    const scaledWidth = width / (currentFile.scale / 100 + 1);
    const scaledHeight = height / (currentFile.scale / 100 + 1);
    switch (ratioMode) {
        case "thin":
            if (imageRatio > 1) {
                return getCanvasSize(ratioMode, scaledHeight);
            } else {
                return {
                    width: scaledWidth,
                    height: (scaledWidth * 5) / 4,
                };
            }
        case "original":
            return { width: scaledWidth, height: scaledWidth / 1.93 };
        case "fat":
            return { width: scaledWidth, height: (scaledWidth * 9) / 16 };
        case "square":
            if (imageRatio > 1) {
                return {
                    width: scaledHeight,
                    height: scaledHeight,
                };
            } else {
                return {
                    width: scaledWidth,
                    height: scaledWidth,
                };
            }
    }
};

const Edit = ({ currentWidth }: EditProps) => {
    const files = useAppSelector((state) => state.upload.files);
    const ratioMode = useAppSelector((state) => state.upload.ratioMode);
    const currentIndex = useAppSelector((state) => state.upload.currentIndex);
    const dispatch = useAppDispatch();
    const [editMode, setEditMode] = useState<"filter1" | "filter2" | "filter3" | "adjust">("filter1");
    const [enteredAdjustInput, setEnteredAdjustInput] =
        useState<null | UploadType.AdjustInputTextType>(null);
    const [imageURLs, setImageURLs] = useState(['loading.gif', 'loading.gif', 'loading.gif', 'loading.gif', 'loading.gif', 'loading.gif', 'loading.gif', 'loading.gif', 'loading.gif', 'loading.gif', 'loading.gif', 'loading.gif']);
    const [originalFile, setOriginalFile] = useState({
        imageRatio: files[currentIndex].imageRatio,
        width: files[currentIndex].width,
        height: files[currentIndex].height,
        url: files[currentIndex].url
    });
    const finalFiles = useAppSelector((state) => state.upload.finalFiles);

    interface ImageData {
        width: number,
        height: number,
    }

    // window 너비에 따라 변경되는 값
    const processedCanvasLayoutWidth = useMemo(
        () => (currentWidth <= MIN_WIDTH ? MIN_WIDTH : currentWidth),
        [currentWidth],
    );
    const canvasSize = useMemo(
        () => getCanvasSize(ratioMode, processedCanvasLayoutWidth),
        [processedCanvasLayoutWidth, ratioMode],
    );

    const adjustInputs: {
        text: UploadType.AdjustInputTextType;
        value: number;
    }[] = useMemo(
        () => [
            { text: "Brightness", value: files[currentIndex].brightness },
            { text: "Contrast", value: files[currentIndex].contrast },
            { text: "Saturation", value: files[currentIndex].saturate },
            { text: "Blur", value: files[currentIndex].blur },
        ],
        [currentIndex, files],
    );

    useEffect(() => {
        if (imageURLs[0] === 'loading.gif') {
            const transformImages = async () => {
                const image = await fetch(files[currentIndex].url);
                const blob = await image.blob();
                const reader = new FileReader();
                reader.onloadend = async () => {
                    const result = reader.result;
                    dispatch(uploadActions.setFinalFiles({
                        relationship: 'original',
                        data: files[currentIndex].url
                    }))

                    const res = await fetch('http://115.145.36.214:8888/filter', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({'image': result, 'gender': 'F'}),
                    });
                    const res_json = await res.json();
                    const urls = [res_json['closest'], res_json['furthest'], res_json['others']].flat();
                    const base64images = urls.map((image) => 'data:image/;base64,' + image);
                    setImageURLs(base64images);
                }
                reader.readAsDataURL(blob);
            };
            transformImages();
        }
    }, [imageURLs]);

    const replaceFile = async (url : string, rel : string) => {
        if (url === 'loading.gif') {
            return;
        }

        // const img = new Image();
        // img.onload = () => {
        //     dispatch(uploadActions.addFile({
        //         imageRatio: img.height / img.width,
        //         width: img.width,
        //         height: img.height,
        //         url: url,
        //     }));
        //     dispatch(uploadActions.prevIndex());
        //     dispatch(uploadActions.deleteFile());
        // };
        // img.src = url;
        dispatch(uploadActions.addFile({
            imageRatio: files[currentIndex].width / files[currentIndex].height,
            width: files[currentIndex].width,
            height: files[currentIndex].height,
            url: url,
        }));
        dispatch(uploadActions.deleteFile());
        console.log(rel, url);
        dispatch(uploadActions.setFinalFiles({
            relationship: rel,
            data: url
        }))
    }

    return (
        <>
            <UploadHeader
                excuteBeforeNextStep={() =>
                    dispatch(uploadActions.resetNewFileUrl())
                }
            />
            <StyledEdit>
                <div
                    className="upload__imgCanvasLayout"
                    style={{
                        width: processedCanvasLayoutWidth + "px",
                        minWidth: processedCanvasLayoutWidth + "px",
                        minHeight: processedCanvasLayoutWidth + "px",
                    }}
                >
                    <EditCanvasUnit
                        currentCanvasWidth={canvasSize.width}
                        currentCanvasHeight={canvasSize.height}
                        currentFile={files[currentIndex]}
                    />
                    {currentIndex > 0 && (
                        <button
                            className="left"
                            onClick={() => dispatch(uploadActions.prevIndex())}
                        >
                            <LeftArrow />
                        </button>
                    )}
                    {currentIndex < files.length - 1 && (
                        <button
                            className="right"
                            onClick={() => dispatch(uploadActions.nextIndex())}
                        >
                            <RightArrow />
                        </button>
                    )}
                </div>
                <div className="upload__imgEditor">
                    <div className="header">
                        <div
                            className={`filter ${
                                editMode === "filter1" ? "active" : ""
                            }`}
                            onClick={() => setEditMode("filter1")}
                        >
                            to Friends
                        </div>
                        <div
                            className={`filter ${
                                editMode === "filter2" ? "active" : ""
                            }`}
                            onClick={() => setEditMode("filter2")}
                        >
                            to Strangers
                        </div>
                        <div
                            className={`filter ${
                                editMode === "filter3" ? "active" : ""
                            }`}
                            onClick={() => setEditMode("filter3")}
                        >
                            to Stalkers
                        </div>
                        <div
                            className={`adjust ${
                                editMode === "adjust" ? "active" : ""
                            }`}
                            onClick={() => setEditMode("adjust")}
                        >
                            Other Fixes
                        </div>
                    </div>
                    {editMode === "adjust" ? (
                        adjustInputs.map((inputObj) => (
                            <div
                                key={inputObj.text}
                                className="adjust__input"
                                onMouseEnter={() =>
                                    setEnteredAdjustInput(inputObj.text)
                                }
                                onMouseLeave={() => setEnteredAdjustInput(null)}
                            >
                                <div>
                                    <div>{inputObj.text}</div>
                                    {inputObj.value !== 0 && (
                                        <button
                                            className={
                                                enteredAdjustInput ===
                                                inputObj.text
                                                    ? "entered"
                                                    : ""
                                            }
                                            onClick={() =>
                                                dispatch(
                                                    uploadActions.resetAdjustInput(
                                                        inputObj.text,
                                                    ),
                                                )
                                            }
                                        >
                                            재설정
                                        </button>
                                    )}
                                </div>
                                <div>
                                    <input
                                        type="range"
                                        onChange={(
                                            event: ChangeEvent<HTMLInputElement>,
                                        ) =>
                                            dispatch(
                                                uploadActions.changeAdjustInput(
                                                    {
                                                        type: inputObj.text,
                                                        value: Number(
                                                            event.target.value,
                                                        ),
                                                    },
                                                ),
                                            )
                                        }
                                        value={inputObj.value}
                                        min={
                                            inputObj.text === "Blur"
                                                ? 0
                                                : -100
                                        }
                                        max="100"
                                        step="1"
                                        style={{
                                            backgroundImage:
                                                inputObj.text !== "Blur"
                                                    ? `linear-gradient(to right, rgb(219, 219, 219) 0%, 
                                    rgb(219, 219, 219) ${Math.min(
                                        50,
                                        inputObj.value / 2 + 50,
                                    )}%, rgb(38, 38, 38) ${Math.min(
                                                          50,
                                                          inputObj.value / 2 +
                                                              50,
                                                      )}%,
                                     rgb(38, 38, 38)   ${Math.max(
                                         50,
                                         inputObj.value / 2 + 50,
                                     )}%, rgb(219, 219, 219)  ${Math.max(
                                                          50,
                                                          inputObj.value / 2 +
                                                              50,
                                                      )}%, rgb(219, 219, 219) 100%)`
                                                    : `linear-gradient(to right, rgb(38, 38, 38) 0%, rgb(38, 38, 38) ${inputObj.value}%, rgb(219, 219, 219) ${inputObj.value}%, rgb(219, 219, 219) 100%)`,
                                        }}
                                    />
                                    <div>{inputObj.value}</div>
                                </div>
                            </div>
                        ))
                    ) : null}
                    {editMode === "filter1" ? (
                        <div className='deepprivacy'>
                            <h1 className='subtitle'>No changes</h1>
                            <div className='image__row'>
                                <img className='image__element' src={originalFile.url} onClick={() => replaceFile(originalFile.url, 'positive')} />
                            </div>
                            
                            <h1 className='subtitle'>DeepPrivacy - Closest Faces</h1>
                            <div className='image__row'>
                                <img className='image__element' src={imageURLs[0]} onClick={() => replaceFile(imageURLs[0], 'positive')} />
                                <img className='image__element' src={imageURLs[1]} onClick={() => replaceFile(imageURLs[1], 'positive')} />
                                <img className='image__element' src={imageURLs[2]} onClick={() => replaceFile(imageURLs[2], 'positive')} />
                            </div>

                            <h1 className='subtitle'>DeepPrivacy - Furthest Faces</h1>
                            <div className='image__row'>
                                <img className='image__element' src={imageURLs[3]} onClick={() => replaceFile(imageURLs[3], 'positive')} />
                                <img className='image__element' src={imageURLs[4]} onClick={() => replaceFile(imageURLs[4], 'positive')} />
                                <img className='image__element' src={imageURLs[5]} onClick={() => replaceFile(imageURLs[5], 'positive')} />
                            </div>

                            <h1 className='subtitle'>Other Transformations</h1>
                            <div className='image__row'>
                                <img className='image__element' src={imageURLs[6]} onClick={() => replaceFile(imageURLs[6], 'positive')} />
                                <img className='image__element' src={imageURLs[7]} onClick={() => replaceFile(imageURLs[7], 'positive')} />
                                <img className='image__element' src={imageURLs[8]} onClick={() => replaceFile(imageURLs[8], 'positive')} />
                            </div>

                            <div className='image__row'>
                                <img className='image__element' src={imageURLs[9]} onClick={() => replaceFile(imageURLs[9], 'positive')} />
                                <img className='image__element' src={imageURLs[10]} onClick={() => replaceFile(imageURLs[10], 'positive')} />
                                <img className='image__element' src={imageURLs[11]} onClick={() => replaceFile(imageURLs[11], 'positive')} />
                            </div>
                            <div style={{height: '24px'}} />
                        </div>
                    ) : null}
                    {editMode === "filter2" ? (
                        <div className='deepprivacy'>
                            <h1 className='subtitle'>No changes</h1>
                            <div className='image__row'>
                                <img className='image__element' src={originalFile.url} onClick={() => replaceFile(originalFile.url, 'neutral')} />
                            </div>

                            <h1 className='subtitle'>DeepPrivacy - Closest Faces</h1>
                            <div className='image__row'>
                                <img className='image__element' src={imageURLs[0]} onClick={() => replaceFile(imageURLs[0], 'neutral')} />
                                <img className='image__element' src={imageURLs[1]} onClick={() => replaceFile(imageURLs[1], 'neutral')} />
                                <img className='image__element' src={imageURLs[2]} onClick={() => replaceFile(imageURLs[2], 'neutral')} />
                            </div>

                            <h1 className='subtitle'>DeepPrivacy - Furthest Faces</h1>
                            <div className='image__row'>
                                <img className='image__element' src={imageURLs[3]} onClick={() => replaceFile(imageURLs[3], 'neutral')} />
                                <img className='image__element' src={imageURLs[4]} onClick={() => replaceFile(imageURLs[4], 'neutral')} />
                                <img className='image__element' src={imageURLs[5]} onClick={() => replaceFile(imageURLs[5], 'neutral')} />
                            </div>

                            <h1 className='subtitle'>Other Transformations</h1>
                            <div className='image__row'>
                                <img className='image__element' src={imageURLs[6]} onClick={() => replaceFile(imageURLs[6], 'neutral')} />
                                <img className='image__element' src={imageURLs[7]} onClick={() => replaceFile(imageURLs[7], 'neutral')} />
                                <img className='image__element' src={imageURLs[8]} onClick={() => replaceFile(imageURLs[8], 'neutral')} />
                            </div>

                            <div className='image__row'>
                                <img className='image__element' src={imageURLs[9]} onClick={() => replaceFile(imageURLs[9], 'neutral')} />
                                <img className='image__element' src={imageURLs[10]} onClick={() => replaceFile(imageURLs[10], 'neutral')} />
                                <img className='image__element' src={imageURLs[11]} onClick={() => replaceFile(imageURLs[11], 'neutral')} />
                            </div>
                            <div style={{height: '24px'}} />
                        </div>
                    ) : null}
                    {editMode === "filter3" ? (
                        <div className='deepprivacy'>
                            <h1 className='subtitle'>No changes</h1>
                            <div className='image__row'>
                                <img className='image__element' src={originalFile.url} onClick={() => replaceFile(originalFile.url, 'negative')} />
                            </div>

                            <h1 className='subtitle'>DeepPrivacy - Closest Faces</h1>
                            <div className='image__row'>
                                <img className='image__element' src={imageURLs[0]} onClick={() => replaceFile(imageURLs[0], 'negative')} />
                                <img className='image__element' src={imageURLs[1]} onClick={() => replaceFile(imageURLs[1], 'negative')} />
                                <img className='image__element' src={imageURLs[2]} onClick={() => replaceFile(imageURLs[2], 'negative')} />
                            </div>

                            <h1 className='subtitle'>DeepPrivacy - Furthest Faces</h1>
                            <div className='image__row'>
                                <img className='image__element' src={imageURLs[3]} onClick={() => replaceFile(imageURLs[3], 'negative')} />
                                <img className='image__element' src={imageURLs[4]} onClick={() => replaceFile(imageURLs[4], 'negative')} />
                                <img className='image__element' src={imageURLs[5]} onClick={() => replaceFile(imageURLs[5], 'negative')} />
                            </div>

                            <h1 className='subtitle'>Other Transformations</h1>
                            <div className='image__row'>
                                <img className='image__element' src={imageURLs[6]} onClick={() => replaceFile(imageURLs[6], 'negative')} />
                                <img className='image__element' src={imageURLs[7]} onClick={() => replaceFile(imageURLs[7], 'negative')} />
                                <img className='image__element' src={imageURLs[8]} onClick={() => replaceFile(imageURLs[8], 'negative')} />
                            </div>

                            <div className='image__row'>
                                <img className='image__element' src={imageURLs[9]} onClick={() => replaceFile(imageURLs[9], 'negative')} />
                                <img className='image__element' src={imageURLs[10]} onClick={() => replaceFile(imageURLs[10], 'negative')} />
                                <img className='image__element' src={imageURLs[11]} onClick={() => replaceFile(imageURLs[11], 'negative')} />
                            </div>
                            <div style={{height: '24px'}} />
                        </div>
                    ) : null}
                </div>
            </StyledEdit>
        </>
    );
};

export default Edit;
