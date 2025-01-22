import styled from "styled-components";

import ContentBox from "components/Common/ContentBox";
import Suggest from "components/Auth/Suggest";
import EmailConfirmForm from "components/Auth/SignUpForm/EmailConfirmForm";
import SignUpForm from "components/Auth/SignUpForm/SignUpFormContent";
import LoginForm from "components/Auth/LoginForm/LoginFormContent";
import { useLocation } from "react-router-dom";
import { useAppSelector } from "app/store/Hooks";

const Container = styled.div<{ pathname: string }>`
    display: flex;
    flex-direction: column;
    min-height: ${(props) => (props.pathname === "/" ? 0 : 100)}vh;
    margin-top: 12px;
    justify-content: center;
    max-width: 350px;
    flex-grow: 1;

    .warning-message {
        padding: 5px;
        text-align: center;
        color: red;

        .warning {
            font-weight: 700;
            margin-right: 5px;
        }

        .not-instagram {
            text-decoration: underline;
        }
    }
`;

const contentBox: UIType.ContentBoxProps = {
    padding: `10px 0`,
    margin: `0 0 10px`,
};

export default function Form(props: { router: "signIn" | "signUp" }) {
    const { pathname } = useLocation();
    const currentForm = useAppSelector((state) => state.auth.currentFormState);

    return (
        <Container pathname={pathname}>
            <ContentBox padding={contentBox.padding} margin={contentBox.margin}>
                {props.router === "signIn" ? (
                    <LoginForm />
                ) : currentForm === "confirmEmail" ? (
                    <EmailConfirmForm />
                ) : (
                    <SignUpForm />
                )}
            </ContentBox>
            <ContentBox padding={contentBox.padding} margin={contentBox.margin}>
                <Suggest />
            </ContentBox>
            <aside className="warning-message">
                <p>
                    <span className="warning">Caution</span>
                    <span className="not-instagram">
                        This is not a real Instagram.
                    </span>
                </p>
                <p>This is a clone page that developers made as side project.</p>
            </aside>
        </Container>
    );
}
